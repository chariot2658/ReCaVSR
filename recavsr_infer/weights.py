"""Strict loading of a single, already-merged DiT safetensors checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .models.transformer_wan import WanRotaryPosEmbed, WanTransformer3DModel
from .runtime.offload import streamed_indices


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


def load_model(
    model_dir: str | Path, *, device="cuda:0", gpu_blocks: int | None = None
) -> WanTransformer3DModel:
    """Load one complete checkpoint strictly; no adapter merging or key rewriting.

    Tensors go straight from the file to the device one at a time, so host
    memory never holds the whole checkpoint. With gpu_blocks, the remaining DiT
    blocks stay on the meta device; TransformerSession materializes each with
    load_block and moves it into the streaming arena (see runtime.offload).
    """
    root = Path(model_dir)
    validate_package(root)
    config = json.loads((root / "model_config.json").read_text())
    model = make_model(config, meta=True)
    streamed = set()
    if gpu_blocks is not None:
        streamed = set(streamed_indices(len(model.blocks), gpu_blocks))
    path = root / "transformer.safetensors"
    with safe_open(path, framework="pt") as stream:
        keys = set(stream.keys())
        expected = set(model.state_dict().keys())
        if keys != expected:
            raise ValueError(
                f"Checkpoint keys differ from the model: missing "
                f"{sorted(expected - keys)[:5]}, unexpected {sorted(keys - expected)[:5]}"
            )
        state = {
            key: stream.get_tensor(key).to(device=device, dtype=torch.bfloat16)
            for key in sorted(keys)
            if _block_index(key) not in streamed
        }
    model.load_state_dict(state, strict=False, assign=True)
    del state
    model = model.eval().requires_grad_(False)
    # Derived buffers (RoPE tables) were built on the CPU.
    blocks, model.blocks = model.blocks, nn.ModuleList()
    model.to(device=device)
    model.blocks = blocks
    model._checkpoint_path = path
    return model


def _block_index(key: str) -> int | None:
    return int(key.split(".")[1]) if key.startswith("blocks.") else None


def load_block(model: WanTransformer3DModel, index: int, *, device) -> None:
    """Materialize one meta-device DiT block from the checkpoint onto device."""
    prefix = f"blocks.{index}."
    with safe_open(model._checkpoint_path, framework="pt") as stream:
        state = {
            key[len(prefix) :]: stream.get_tensor(key).to(
                device=device, dtype=torch.bfloat16
            )
            for key in stream.keys()
            if key.startswith(prefix)
        }
    model.blocks[index].load_state_dict(state, strict=True, assign=True)
    model.blocks[index].requires_grad_(False)
