"""Optional LR-conditioned FlashDecoder with explicit per-video streaming state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..models.flashdecoder.flashdecoder_wan22 import FlashDecoder, FlashDecoderState
from .compat import inductor_options


def load_flashdecoder(vae_config, checkpoint_path, *, device="cuda:0"):
    """Load the main-model-only safetensors and its adjacent inference config."""
    if checkpoint_path is None:
        raise ValueError("FlashDecoder requires a flashdecoder.safetensors file.")
    path = Path(checkpoint_path)
    config = json.loads((path.parent / "flashdecoder_config.json").read_text())
    if (
        len(vae_config.get("latents_mean", [])) != 48
        or len(vae_config.get("latents_std", [])) != 48
        or vae_config.get("patch_size") != 2
    ):
        raise ValueError("FlashDecoder requires Wan2.2 48-channel normalization.")
    if (
        set(config) != {"variant", "window_size", "spatial_window_size"}
        or config["variant"] != "S"
    ):
        raise ValueError(
            "Expected the released streaming FlashDecoder-S configuration."
        )
    # CPU construction preserves nonpersistent RoPE buffers. Retain FP32 master
    # parameters; the session applies BF16 autocast, not a permanent weight cast.
    model = FlashDecoder.from_variant(**config)
    weights = load_file(path)
    if any(t.dtype != torch.float32 for t in weights.values()):
        raise ValueError("FlashDecoder weights must retain their original FP32 dtype.")
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model = model.to(device=device).eval().requires_grad_(False)
    model.decoder_type = "flashdecoder"
    model.vae_config = dict(vae_config)
    model.decoder_source = {
        "type": "flashdecoder",
        "checkpoint": str(path.resolve()),
        "config": config,
    }
    return model


class FlashDecoderSession:
    """Normalized DiT latents + target-resolution LQ -> [-1,1] RGB chunks."""

    def __init__(self, model, *, compile_decoder=False, compile_mode="default"):
        self.model = model
        self.state = model.init_state()
        device = next(model.parameters()).device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.mean = torch.tensor(
            model.vae_config["latents_mean"], device=device, dtype=dtype
        ).view(1, 48, 1, 1, 1)
        self.std = torch.tensor(
            model.vae_config["latents_std"], device=device, dtype=dtype
        ).view(1, 48, 1, 1, 1)
        self.compile_decoder = compile_decoder
        self.decode_core = self._tensor_step
        if compile_decoder:
            if device.type != "cuda":
                raise ValueError("Compiled FlashDecoder requires CUDA.")
            if compile_mode not in ("default", "max-autotune-no-cudagraphs"):
                raise ValueError(
                    "FlashDecoder compilation requires a non-CUDA-graph mode."
                )
            from .flash_norm import preserve_native_norms

            preserve_native_norms(model)
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64
            )
            self.decode_core = torch.compile(
                self._tensor_step,
                fullgraph=True,
                dynamic=False,
                options=inductor_options(
                    {
                        **torch._inductor.list_mode_options(compile_mode),
                        "emulate_precision_casts": True,
                        "force_same_precision": True,
                        # No "freezing": AOTAutograd cannot cache frozen graphs, so
                        # every process recompiled ~20 s. Output is byte-identical.
                        "comprehensive_padding": False,
                    }
                ),
            )

    def reset(self):
        """Start a new video; compiled steps are kept."""
        self.state = self.model.init_state()

    @staticmethod
    def _tensor_step(model, latent_frame, lr_up, backbone, refinement, first):
        # The growing chronological counter must never enter a compiled graph.
        batch, _, height, width = latent_frame.shape
        state = FlashDecoderState(
            backbone=backbone,
            refinement=refinement,
            frame_index=0 if first else 1,
            batch_size=batch,
            latent_height=height,
            latent_width=width,
            spatial_window_size=model.spatial_window_size,
        )
        rgb, next_state = model.decode_step(latent_frame, state, lr_up=lr_up)
        return rgb, next_state.backbone, next_state.refinement

    def _run_compiled_step(self, latent_frame, lr_up):
        rgb, backbone, refinement = self.decode_core(
            self.model,
            latent_frame,
            lr_up,
            self.state.backbone,
            self.state.refinement,
            self.state.frame_index == 0,
        )
        self.state = replace(
            self.state,
            backbone=backbone,
            refinement=refinement,
            frame_index=self.state.frame_index + 1,
            batch_size=latent_frame.shape[0],
            latent_height=latent_frame.shape[2],
            latent_width=latent_frame.shape[3],
        )
        return rgb

    @torch.inference_mode()
    def step(self, latents, *, lr_up=None):
        if latents.ndim != 5 or latents.shape[1] != 48 or min(latents.shape) < 1:
            raise ValueError("FlashDecoder expects nonempty [B,48,T,H,W] latents.")
        if not latents.is_floating_point() or latents.device != self.mean.device:
            raise ValueError("Latents must be floating-point on the decoder device.")
        expected_frames = 4 * latents.shape[2] - (
            3 if self.state.frame_index == 0 else 0
        )
        self.model._validate_lr(latents, lr_up, pixel_frames=expected_frames)
        self.model._validate_state(
            self.state,
            batch_size=latents.shape[0],
            height=latents.shape[3],
            width=latents.shape[4],
        )
        offset = 0
        for i in range(latents.shape[2]):
            take = 1 if self.state.frame_index == 0 else 4
            raw = latents[:, :, i : i + 1].to(self.mean.dtype) * self.std + self.mean
            with torch.autocast(
                device_type=raw.device.type,
                dtype=torch.bfloat16,
                enabled=raw.device.type == "cuda",
            ):
                if self.compile_decoder:
                    rgb = self._run_compiled_step(
                        raw[:, :, 0], lr_up[:, :, offset : offset + take]
                    )
                else:
                    rgb, self.state = self.model.decode_step(
                        raw[:, :, 0],
                        self.state,
                        lr_up=lr_up[:, :, offset : offset + take],
                    )
            offset += take
            yield rgb.clamp(-1, 1)


def decode_block(session, latents, lq):
    """Pass aligned LR only to the conditional decoder; preserve existing decoders."""
    if isinstance(session, FlashDecoderSession):
        return session.step(latents, lr_up=lq)
    return session.step(latents)
