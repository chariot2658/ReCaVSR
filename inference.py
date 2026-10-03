"""Offline, single-GPU, model-streaming ReCaVSR inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from recavsr_infer.color_fix import adain_color_fix, wavelet_color_fix
from recavsr_infer.runtime.constants import (
    DEFAULT_SPATIAL_WINDOW,
    ROPE_WINDOW_LATENTS,
    SPATIAL_TOKEN_STRIDE,
)
from recavsr_infer.runtime.decoder import create_decoder_session, load_decoder
from recavsr_infer.runtime.flashdecoder import decode_block
from recavsr_infer.runtime.transformer import TransformerSession
from recavsr_infer.video import (
    block_schedule,
    prepare_block,
    read_video,
    rgb_uint8,
    scaled_size,
    validate_scale,
    video_pixel_format,
    write_video,
)
from recavsr_infer.weights import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Model directory (default: checkpoints).",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Spatial SR scale (default: 4). Positive decimals supported; dimensions round half-up.",
    )
    parser.add_argument(
        "--color-fix",
        choices=("none", "adain", "wavelet"),
        default="none",
        help="Framewise color correction using matching LQ frames (default: none).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decoder", choices=("wan", "flashdecoder"), default="flashdecoder"
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        help="FlashDecoder main-model safetensors (default: MODEL_DIR/flashdecoder.safetensors).",
    )

    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        default=DEFAULT_SPATIAL_WINDOW,
        metavar=("H", "W"),
        help="Self-attention window in spatial tokens, not pixels (default: 22 40).",
    )
    parser.add_argument(
        "--compile-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile DiT blocks with torch.compile (default: enabled).",
    )
    parser.add_argument(
        "--compile-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile Wan or FlashDecoder (default: enabled).",
    )
    parser.add_argument(
        "--channels-last-decoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wan Conv3d layout (default: enabled for Wan, disabled for FlashDecoder).",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        help="DiT compilation mode only; VAE compilation uses the default mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args(argv)
    if args.decoder == "flashdecoder" and args.decoder_checkpoint is None:
        args.decoder_checkpoint = args.model_dir / "flashdecoder.safetensors"
    if args.channels_last_decoder is None:
        args.channels_last_decoder = args.decoder == "wan"
    if args.decoder == "flashdecoder":
        if args.channels_last_decoder:
            parser.error(
                "FlashDecoder does not use the Wan Conv3d channels-last layout."
            )
    elif args.decoder_checkpoint is not None:
        parser.error("--decoder-checkpoint only applies to flashdecoder.")

    try:
        args.scale = validate_scale(args.scale)
    except ValueError as error:
        parser.error(str(error))
    if min(args.window) < 1:
        parser.error("Spatial windows must be positive.")
    if args.compile_mode != "default" and not args.compile_blocks:
        parser.error("--compile-mode requires --compile-blocks.")
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    if any(path.exists() or path.is_symlink() for path in (args.output, report_path)):
        parser.error("Output or its report already exists; choose a new output path.")
    if args.output.suffix.lower() != ".mp4":
        parser.error("Output must end in .mp4.")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive.")
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive.")
    return args


def save_result(
    output_path: Path, frames: np.ndarray, rate, report: dict[str, Any]
) -> None:
    """Encode locally, then publish without overwriting existing results.

    A failed copy removes only files created by this call, so a partial video
    does not look like a completed result on the next run.
    """
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recavsr-output-") as scratch:
        staged = Path(scratch) / "output.mp4"
        write_video(staged, frames, rate)
        created = []
        try:
            # Reserve both names before copying; 'x' also rejects broken symlinks.
            with output_path.open("xb") as destination:
                created.append((output_path, os.fstat(destination.fileno())))
                with report_path.open("x", encoding="utf-8") as report_stream:
                    created.append((report_path, os.fstat(report_stream.fileno())))
                    with staged.open("rb") as source:
                        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                    json.dump(report, report_stream, indent=2)
        except BaseException:
            for path, identity in reversed(created):
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == (
                        identity.st_dev,
                        identity.st_ino,
                    ):
                        path.unlink()
                except FileNotFoundError:
                    pass
            raise


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This runtime requires an available CUDA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {device.index}")
    frames, rate = read_video(args.input, fps=args.fps, limit=args.frames)
    height, width = scaled_size(*frames.shape[1:3], args.scale)
    # CPU I/O is intentionally offline; only the model and its caches stream.
    # Keep both sessions alive across all chunks of this video.
    with torch.inference_mode(), torch.cuda.device(device):
        model = load_model(args.model_dir, device=device)
        vae = load_decoder(
            args.model_dir,
            device=device,
            channels_last=args.channels_last_decoder,
            decoder=args.decoder,
            checkpoint_path=args.decoder_checkpoint,
        )
        prompt = load_file(args.model_dir / "prompt.safetensors")["positive"]
        transformer = TransformerSession(
            model,
            prompt,
            height + (-height) % SPATIAL_TOKEN_STRIDE,
            width + (-width) % SPATIAL_TOKEN_STRIDE,
            seed=args.seed,
            window=args.window,
            compile_blocks=args.compile_blocks,
            compile_mode=args.compile_mode,
        )
        decoder = create_decoder_session(vae, compile_decoder=args.compile_decoder)
        output = np.empty((len(frames), height, width, 3), dtype=np.uint8)
        events = []
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        for start, valid, block in block_schedule(len(frames)):
            lq = prepare_block(
                frames, start, valid, block, device=device, scale=args.scale
            )
            dit_start, dit_end, decode_end = [
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            ]
            dit_start.record()
            latent = transformer.step(lq)
            dit_end.record()
            remaining, offset = valid, start
            for decoded in decode_block(decoder, latent, lq):
                take = min(remaining, decoded.shape[2])
                if take:
                    rgb = decoded[:, :, :take, :height, :width]
                    if args.color_fix == "wavelet":
                        output[offset : offset + take] = wavelet_color_fix(
                            rgb, frames[offset : offset + take]
                        )
                    else:
                        if args.color_fix == "adain":
                            # Decoder chunks are shorter than DiT chunks; align by
                            # their absolute frame offset and exclude spatial padding.
                            reference_start = offset - start
                            reference = lq[
                                :,
                                :,
                                reference_start : reference_start + take,
                                :height,
                                :width,
                            ]
                            rgb = adain_color_fix(rgb, reference)
                        output[offset : offset + take] = rgb_uint8(rgb, height, width)
                    offset += take
                    remaining -= take
            decode_end.record()
            events.append((dit_start, dit_end, decode_end))
            if remaining:
                raise RuntimeError("Decoder returned too few frames.")
            print(f"frames {start + valid}/{len(frames)}", flush=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        report = {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "frame_count": len(output),
            "height": height,
            "width": width,
            "scale": args.scale,
            "decoder_source": getattr(
                vae, "decoder_source", {"type": "wan", "source": "model package"}
            ),
            "input_height": frames.shape[1],
            "input_width": frames.shape[2],
            "size_rounding": "round-half-up independently per dimension",
            "pixel_format": video_pixel_format(height, width),
            "fps": str(rate),
            "settings": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "model_dir": str(args.model_dir.resolve()),
            "decoder": args.decoder,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "rolling_rope": {
                "interval": ROPE_WINDOW_LATENTS,
                "origin": "max(0, current_start + current_frames - 22)",
            },
            "model_loop_seconds_including_preprocessing_and_cpu_copy": elapsed,
            "dit_ms_including_first_use_compile": sum(
                start.elapsed_time(end) for start, end, _ in events
            ),
            "decode_color_fix_and_cpu_copy_ms_including_first_use_compile": sum(
                end.elapsed_time(decoded) for _, end, decoded in events
            ),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "audio": "not copied",
            "frame_rate_policy": "constant average input fps",
        }
    save_result(args.output, output, rate, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "settings"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
