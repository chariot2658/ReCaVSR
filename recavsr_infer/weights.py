"""Strict loading of a single, already-merged DiT safetensors checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .models.transformer_wan import WanRotaryPosEmbed, WanTransformer3DModel


def make_model(config: dict, *, meta: bool = False) -> WanTransformer3DModel:
    """Build the checkpoint architecture; optional meta initialization avoids duplicate weights."""
    with torch.device("meta" if meta else "cpu"):
        model = WanTransformer3DModel.from_config(config)
        model.lq_proj = model.build_lq_proj()
        model.sr_proj = model.build_sr_proj()
    if meta:
        # RoPE tables are derived, nonpersistent buffers, absent from safetensors.
        # Rebuild them off the meta device before loading checkpoint parameters.
        model.rope = WanRotaryPosEmbed(
            model.config.attention_head_dim,
            model.config.patch_size,
            model.config.rope_max_seq_len,
        )
    return model


def validate_package(root: str | Path) -> None:
    """Check shared inference assets; decoder-specific weights are loaded separately."""
    root = Path(root)
    required = (
        "transformer.safetensors",
        "model_config.json",
        "vae_config.json",
        "prompt.safetensors",
    )
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty model file: {path}")
    # Inspect tensor headers before allocating the checkpoint.
    with safe_open(root / "transformer.safetensors", framework="pt") as stream:
        keys = list(stream.keys())
    if any(".lora_" in key or key.endswith((".lora_A", ".lora_B")) for key in keys):
        raise ValueError(
            "Adapter tensors are not supported; use a pre-merged checkpoint."
        )
    for prefix in ("blocks.", "lq_proj.", "sr_proj."):
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(f"DiT checkpoint is missing {prefix} parameters.")


def load_model(model_dir: str | Path, *, device="cuda:0") -> WanTransformer3DModel:
    """Load one complete checkpoint strictly; no adapter merging or key rewriting."""
    root = Path(model_dir)
    validate_package(root)
    config = json.loads((root / "model_config.json").read_text())
    model = make_model(config, meta=True)
    state = load_file(root / "transformer.safetensors")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
