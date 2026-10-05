"""Single-GPU, streaming ReCaVSR inference with optional DiT block offload."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

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
from recavsr_infer.runtime.offload import plan_gpu_blocks
from recavsr_infer.runtime.transformer import TransformerSession
from recavsr_infer.strips import blend_strips, strip_columns
from recavsr_infer.video import (
    BackgroundWriter,
    VideoWriter,
    block_tensor,
    enlarge_block,
    iter_blocks,
    open_video,
    prefetch,
    rgb_uint8,
    scaled_size,
    validate_scale,
    video_pixel_format,
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
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--fp8-dit",
        action="store_true",
        help="Use lossy FP8 DiT linear calculations; attention and decoder keep their precision.",
    )
    precision.add_argument(
        "--nvfp4-dit",
        action="store_true",
        help="Use faster, lossy NVFP4 DiT linears on Blackwell; attention and decoder keep their precision.",
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
    parser.add_argument(
        "--output-size",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="Exact output raster; overrides --scale (e.g. 1920 1080 for anamorphic DVD).",
    )
    parser.add_argument(
        "--gpu-blocks",
        default="auto",
        help="DiT blocks kept on the GPU; the rest stream from pinned host memory "
        "(default: auto, from free VRAM minus --vram-reserve).",
    )
    parser.add_argument(
        "--vram-reserve",
        type=float,
        default=3.0,
        help="GiB left free for activations and the decoder when --gpu-blocks=auto.",
    )
    parser.add_argument(
        "--strips",
        type=int,
        default=1,
        help="Run the frame as N overlapping vertical strips, one full-length "
        "session each, then feather-blend (for rasters too large for VRAM).",
    )
    parser.add_argument(
        "--strip-overlap",
        type=int,
        default=48,
        help="Strip overlap in input (LQ) pixels (default: 48).",
    )
    parser.add_argument("--only-strip", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="With --strips: stop after writing OUTPUT_STEM.stripK.mp4, leaving the "
        "blend to the caller (e.g. while the GPU runs the next video).",
    )
    parser.add_argument(
        "--strip-crf",
        type=int,
        default=8,
        help="libx264 CRF of the intermediate strip videos (default: 8).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="Area-downscale the final frames to W x H (e.g. 4x then 1920 1080).",
    )
    parser.add_argument(
        "--crf", type=int, default=18, help="libx264 CRF (default: 18)."
    )
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
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
    if args.output_size is not None and min(args.output_size) < 1:
        parser.error("--output-size must be positive.")
    if args.strips < 1 or args.strip_overlap < 1:
        parser.error("--strips and --strip-overlap must be positive.")
    if args.no_blend and args.strips < 2:
        parser.error("--no-blend requires --strips.")
    if args.strips > 1 and args.output_size is not None:
        parser.error("--strips uses --scale; set the final size with --resize.")
    if args.resize is not None and min(args.resize) < 1:
        parser.error("--resize must be positive.")
    if args.gpu_blocks != "auto":
        try:
            args.gpu_blocks = int(args.gpu_blocks)
        except ValueError:
            parser.error("--gpu-blocks must be 'auto' or an integer.")
        if args.gpu_blocks < 0:
            parser.error("--gpu-blocks must be non-negative.")
    if min(args.window) < 1:
        parser.error("Spatial windows must be positive.")
    if (
        os.name == "nt"
        and not sys.flags.utf8_mode
        and (args.compile_blocks or args.compile_decoder)
    ):
        # Inductor reads its templates with the locale codec (cp950/cp1252 fail).
        parser.error(
            "torch.compile on Windows needs Python UTF-8 mode: set PYTHONUTF8=1, "
            "or pass --no-compile-blocks --no-compile-decoder."
        )
    if args.compile_mode != "default" and not args.compile_blocks:
        parser.error("--compile-mode requires --compile-blocks.")
    if (args.fp8_dit or args.nvfp4_dit) and not args.compile_blocks:
        parser.error("Quantized DiT modes require --compile-blocks for efficient activation quantization.")
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


def publish(staged: Path, final: Path) -> None:
    """Rename a finished file into place; a killed run never leaves a final name."""
    os.replace(staged, final)


def report_memory(device, stats) -> None:
    stats["peak_allocated"] = max(
        stats["peak_allocated"], torch.cuda.max_memory_allocated(device)
    )
    stats["peak_reserved"] = max(
        stats["peak_reserved"], torch.cuda.max_memory_reserved(device)
    )


def run_video(blocks, writer, transformer, decoder, args, device, height, width, stats):
    """Stream one video's blocks through the DiT and decoder into writer.

    Decoding and upsampling the next block run on a background thread, so the
    GPU only waits for the host-to-device copy.
    """
    start_time = time.perf_counter()
    staged = (
        (
            start,
            chunk,
            block,
            enlarge_block(chunk, (height, width), block=block, pin=True),
        )
        for start, chunk, block in blocks
    )
    for start, chunk, block, enlarged in prefetch(staged):
        valid = len(chunk)
        lq = block_tensor(enlarged, block, device=device)
        dit_start, dit_end, decode_end = [
            torch.cuda.Event(enable_timing=True) for _ in range(3)
        ]
        dit_start.record()
        latent = transformer.step(lq)
        dit_end.record()
        remaining, offset = valid, 0
        for decoded in decode_block(decoder, latent, lq):
            take = min(remaining, decoded.shape[2])
            if take:
                rgb = decoded[:, :, :take, :height, :width]
                if args.color_fix == "wavelet":
                    out = wavelet_color_fix(rgb, chunk[offset : offset + take])
                else:
                    if args.color_fix == "adain":
                        # Decoder chunks are shorter than DiT chunks; align by
                        # their offset in the block and exclude spatial padding.
                        reference = lq[:, :, offset : offset + take, :height, :width]
                        rgb = adain_color_fix(rgb, reference)
                    out = rgb_uint8(rgb, height, width)
                writer.write(out)
                offset += take
                remaining -= take
        decode_end.record()
        stats["events"].append((dit_start, dit_end, decode_end))
        if remaining:
            raise RuntimeError("Decoder returned too few frames.")
        done = start + valid
        elapsed = time.perf_counter() - start_time
        step_peak = torch.cuda.max_memory_allocated(device)
        report_memory(device, stats)
        torch.cuda.reset_peak_memory_stats(device)
        print(
            f"{stats['label']}frames {done} | {elapsed / done:.3f} s/frame | step peak "
            f"{step_peak / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.memory_reserved(device) / 2**30:.2f} GiB reserved",
            flush=True,
        )
    return writer.frames


def write_published(path: Path, frames, rate, height, width, *, crf, preset) -> int:
    """Encode an iterator of RGB frames to path via a .part file."""
    staged = path.with_name(path.stem + ".part" + path.suffix)
    writer = VideoWriter(staged, rate, height, width, crf=crf, preset=preset)
    try:
        for frame in frames:
            writer.write(frame[None])
        writer.close()
    except BaseException:
        writer.output.close()
        staged.unlink(missing_ok=True)
        raise
    publish(staged, path)
    return writer.frames


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This runtime requires an available CUDA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {device.index}")
    if args.fp8_dit and torch.cuda.get_device_capability(device) < (8, 9):
        raise ValueError("FP8 DiT requires an NVIDIA GPU with compute capability 8.9 or later.")
    if args.nvfp4_dit:
        from recavsr_infer.runtime.nvfp4 import validate_support

        validate_support(device)
    probe, rate = open_video(args.input, fps=args.fps, limit=1)
    input_height, input_width = next(probe).shape[:2]
    columns = strip_columns(input_width, args.strips, args.strip_overlap, args.scale)
    strip_width = columns[0][1] - columns[0][0]
    if args.output_size is not None:
        width, height = args.output_size
    else:
        height, width = scaled_size(input_height, strip_width, args.scale)
    full_height, full_width = (
        scaled_size(input_height, input_width, args.scale)
        if args.strips > 1
        else (height, width)
    )
    final_width, final_height = args.resize or (full_width, full_height)
    model_height = height + (-height) % SPATIAL_TOKEN_STRIDE
    model_width = width + (-width) % SPATIAL_TOKEN_STRIDE
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    precision_suffix = ".nvfp4" if args.nvfp4_dit else ".fp8" if args.fp8_dit else ""
    strip_paths = [
        args.output.with_name(
            f"{args.output.stem}{precision_suffix}.strip{i}{args.output.suffix}"
        )
        for i in range(len(columns))
    ]
    stats = {"events": [], "peak_allocated": 0, "peak_reserved": 0, "label": ""}
    start_time = time.perf_counter()
    offload_plan, streamed, model_blocks = None, [], 0
    todo = [
        i for i, path in enumerate(strip_paths) if args.strips == 1 or not path.exists()
    ]
    if args.strips > 1 and args.only_strip is None:
        # One process per strip: each starts with an unfragmented CUDA allocator
        # (in-process resets leave pinned segments that push later strips past
        # free VRAM into slow shared memory).
        for i in todo:
            command = [sys.executable, *sys.argv, "--only-strip", str(i)]
            code = subprocess.run(command).returncode
            if code:
                raise SystemExit(code)
        todo = []
    elif args.only_strip is not None:
        todo = [i for i in todo if i == args.only_strip]
    with torch.inference_mode(), torch.cuda.device(device):
        if todo:
            vae = load_decoder(
                args.model_dir,
                device=device,
                channels_last=args.channels_last_decoder,
                decoder=args.decoder,
                checkpoint_path=args.decoder_checkpoint,
            )
            gpu_blocks = args.gpu_blocks
            if gpu_blocks == "auto":
                gpu_blocks, offload_plan = plan_gpu_blocks(
                    args.model_dir,
                    device=device,
                    height=model_height,
                    width=model_width,
                    reserve_bytes=int(args.vram_reserve * 2**30),
                )
                print("offload plan:", json.dumps(offload_plan), flush=True)
            model = load_model(args.model_dir, device=device, gpu_blocks=gpu_blocks)
            prompt = load_file(args.model_dir / "prompt.safetensors")["positive"]
            transformer = TransformerSession(
                model,
                prompt,
                model_height,
                model_width,
                seed=args.seed,
                window=args.window,
                compile_blocks=args.compile_blocks,
                compile_mode=args.compile_mode,
                fp8_linears=args.fp8_dit,
                nvfp4_linears=args.nvfp4_dit,
            )
            streamer = transformer.streamer
            streamed, model_blocks = streamer.order, len(model.blocks)
            print(
                f"DiT blocks: {model_blocks - len(streamed)} resident, "
                f"{len(streamed)} streamed; pinned host "
                f"{streamer.host_bytes / 2**30:.2f} GiB, slots "
                f"{streamer.slot_bytes / 2**30:.2f} GiB; "
                f"allocated {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB",
                flush=True,
            )
            decoder = create_decoder_session(vae, compile_decoder=args.compile_decoder)
            torch.cuda.reset_peak_memory_stats(device)
        for n, i in enumerate(todo):
            if n:
                transformer.reset(args.seed)
                if hasattr(decoder, "reset"):
                    decoder.reset()
                else:
                    decoder = create_decoder_session(
                        vae, compile_decoder=args.compile_decoder
                    )
                # Return the previous strip's cached blocks to the driver; otherwise
                # the new strip's caches are allocated beside them and spill.
                torch.cuda.empty_cache()
                print(
                    f"strip {i + 1} reset: allocated "
                    f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB, reserved "
                    f"{torch.cuda.memory_reserved(device) / 2**30:.2f} GiB",
                    flush=True,
                )
            x0, x1 = columns[i]
            frames, _ = open_video(args.input, fps=args.fps, limit=args.frames)
            if args.strips > 1:
                frames = (np.ascontiguousarray(f[:, x0:x1]) for f in frames)
                stats["label"] = f"strip {i + 1}/{len(columns)} "
                target, crf, preset = strip_paths[i], args.strip_crf, "veryfast"
            else:
                target, crf, preset = args.output, args.crf, args.preset
            staged = target.with_name(target.stem + ".part" + target.suffix)
            writer = BackgroundWriter(
                VideoWriter(staged, rate, height, width, crf=crf, preset=preset)
            )
            try:
                run_video(
                    iter_blocks(frames),
                    writer,
                    transformer,
                    decoder,
                    args,
                    device,
                    height,
                    width,
                    stats,
                )
                writer.close()
            except BaseException:
                writer.abort()
                staged.unlink(missing_ok=True)
                raise
            if args.strips == 1 and args.resize is not None:
                resized = args.output.with_name(args.output.stem + ".full.mp4")
                publish(staged, resized)
            else:
                publish(staged, target)
        if todo:
            torch.cuda.synchronize(device)
    gpu_seconds = time.perf_counter() - start_time
    if args.only_strip is not None or args.no_blend:
        return
    if args.strips > 1:
        print(f"blending {len(columns)} strips", flush=True)
        frames = blend_strips(
            strip_paths, columns, args.scale, full_height, full_width, args.resize
        )
        frame_count = write_published(
            args.output,
            frames,
            rate,
            final_height,
            final_width,
            crf=args.crf,
            preset=args.preset,
        )
        for path in strip_paths:
            path.unlink()
    elif args.resize is not None:
        full = args.output.with_name(args.output.stem + ".full.mp4")
        frames = blend_strips(
            [full], [(0, input_width)], width / input_width, height, width, args.resize
        )
        frame_count = write_published(
            args.output,
            frames,
            rate,
            final_height,
            final_width,
            crf=args.crf,
            preset=args.preset,
        )
        full.unlink()
    else:
        frame_count = None
    events = stats["events"]
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "frame_count": frame_count,
        "height": final_height,
        "width": final_width,
        "model_height": height,
        "model_width": width,
        "scale": args.scale,
        "output_size": args.output_size,
        "strips": columns,
        "input_height": input_height,
        "input_width": input_width,
        "size_rounding": "round-half-up independently per dimension",
        "pixel_format": video_pixel_format(final_height, final_width),
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
        "offload": {
            "plan": offload_plan,
            "resident_blocks": model_blocks - len(streamed),
            "streamed_blocks": streamed,
        },
        "seconds_including_blend": time.perf_counter() - start_time,
        "model_seconds_this_run": gpu_seconds,
        "dit_ms_including_first_use_compile": sum(
            start.elapsed_time(end) for start, end, _ in events
        ),
        "decode_color_fix_and_cpu_copy_ms_including_first_use_compile": sum(
            end.elapsed_time(decoded) for _, end, decoded in events
        ),
        "peak_allocated_gib": stats["peak_allocated"] / 2**30,
        "peak_reserved_gib": stats["peak_reserved"] / 2**30,
        "audio": "not copied",
        "frame_rate_policy": "constant average input fps",
    }
    staged_report = report_path.with_name(report_path.name + ".part")
    staged_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    publish(staged_report, report_path)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "settings"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
