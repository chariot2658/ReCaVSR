"""Per-video causal Wan decoder; completed RGB chunks need not remain on GPU."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from ..models.autoencoder_kl_wan import AutoencoderKLWan, unpatchify
from .compat import inductor_options


def load_decoder(
    model_dir,
    *,
    device="cuda:0",
    channels_last=False,
    decoder="wan",
    checkpoint_path=None,
):
    if decoder not in ("wan", "flashdecoder"):
        raise ValueError(f"Unknown decoder: {decoder}")
    if decoder == "wan" and checkpoint_path is not None:
        raise ValueError(
            "Wan uses the model package, not a separate decoder checkpoint."
        )
    if decoder == "flashdecoder" and channels_last:
        raise ValueError("channels_last is only supported by the Wan decoder.")
    root = Path(model_dir)
    config = json.loads((root / "vae_config.json").read_text())
    if decoder == "flashdecoder":
        from .flashdecoder import load_flashdecoder

        return load_flashdecoder(config, checkpoint_path, device=device)
    with torch.device("meta"):
        model = AutoencoderKLWan.from_config(config)
    model.load_state_dict(load_file(root / "vae.safetensors"), strict=True, assign=True)
    # Inference is decoder-only, but strict validation covers the complete checkpoint.
    model.encoder, model.quant_conv = nn.Identity(), nn.Identity()
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if channels_last:
        # Set the weight layout before creating any compiled decoder session.
        for module in model.modules():
            if isinstance(module, nn.Conv3d):
                module.weight = nn.Parameter(
                    module.weight.detach().contiguous(
                        memory_format=torch.channels_last_3d
                    ),
                    requires_grad=False,
                )
    return model


def create_decoder_session(model, *, compile_decoder=False, compile_mode="default"):
    if getattr(model, "decoder_type", "wan") == "flashdecoder":
        from .flashdecoder import FlashDecoderSession

        return FlashDecoderSession(
            model, compile_decoder=compile_decoder, compile_mode=compile_mode
        )
    return DecoderSession(
        model, compile_decoder=compile_decoder, compile_mode=compile_mode
    )


class DecoderSession:
    """Causal decoder cache for one video; keep it alive across DiT steps."""

    def __init__(self, model, *, compile_decoder=False, compile_mode="default"):
        self.model = model
        parameter = next(model.parameters())
        self.mean = torch.tensor(
            model.config.latents_mean, device=parameter.device, dtype=parameter.dtype
        ).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(
            model.config.latents_std, device=parameter.device, dtype=parameter.dtype
        ).view(1, -1, 1, 1, 1)
        self.history = (None,) * model._cached_conv_counts["decoder"]
        self.first = True
        self.decode_chunk = model._decode_chunk_with_cache
        if compile_decoder:
            self.decode_chunk = torch.compile(
                self.decode_chunk,
                fullgraph=True,
                dynamic=False,
                options=inductor_options(
                    {
                        **torch._inductor.list_mode_options(compile_mode),
                        "emulate_precision_casts": True,
                        "force_same_precision": True,
                        # Preserve the validated layout for cached causal Conv3d decoding.
                        "comprehensive_padding": False,
                    }
                ),
            )

    @torch.inference_mode()
    def step(self, latents):
        """Yield [1,3,1/4,H,W] tensors; history persists across DiT blocks."""
        # The first latent emits one RGB frame; subsequent latents emit four.
        for i in range(latents.shape[2]):
            raw = latents[:, :, i : i + 1].to(self.mean.dtype) * self.std + self.mean
            raw = self.model.post_quant_conv(raw)
            decoded, history, _ = self.decode_chunk(raw, self.history, self.first)
            self.history = history
            self.first = False
            if self.model.config.patch_size is not None:
                decoded = unpatchify(decoded, self.model.config.patch_size)
            yield decoded.clamp(-1, 1)
