"""Lossless DiT block streaming for GPUs that cannot hold every block.

Streamed blocks keep their weights in one page-locked host arena and borrow a
GPU slot buffer while they run. Their parameters are views into the slot, so a
compiled block sees fixed addresses; only the slot contents change. Copies run
on a side stream, ordered cyclically, so each transfer overlaps the compute of
the blocks scheduled before it (including the previous step's tail and decode).
"""

from __future__ import annotations

import torch
from torch import nn

SLOTS = 2


def streamed_indices(total: int, resident: int, slots: int = SLOTS) -> list[int]:
    """Spread streamed blocks evenly so every copy has compute to hide behind.

    When blocks rotate through slots, their count is rounded up to a multiple of
    the slot count so each block keeps one fixed slot across steps.
    """
    streamed = max(0, min(total, total - resident))
    if streamed > slots:
        streamed = min(total - total % slots, -(-streamed // slots) * slots)
    return sorted({int((k + 0.5) * total / streamed) for k in range(streamed)})


def drop_cross_attention_kv(block: nn.Module) -> None:
    """Free cross-attention K/V projections once their output is precomputed."""
    a = block.attn2
    a.to_k = a.to_v = None
    if getattr(a, "to_kv", None) is not None:
        a.to_kv = None


ALIGN = 64  # Elements; keeps every slot view 128-byte aligned for Triton kernels.


def _offsets(params) -> tuple[list[int], int]:
    offsets, total = [], 0
    for _, p in params:
        offsets.append(total)
        total += -(-p.numel() // ALIGN) * ALIGN
    return offsets, total


def forward_parameters(block: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in block.named_parameters() if p is not None]


def forward_bytes(block: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for _, p in forward_parameters(block))


def page_locked(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate exactly-sized page-locked host memory.

    pin_memory=True goes through the caching host allocator, which rounds each
    request up to a power of two (a 0.27 GiB block pins 0.5 GiB); registering
    one ordinary allocation pins only what is used.
    """
    host = torch.zeros(numel, dtype=dtype)
    error = torch.cuda.cudart().cudaHostRegister(
        host.data_ptr(), host.numel() * host.element_size(), 0
    )
    if int(error) != 0:
        raise RuntimeError(f"cudaHostRegister failed ({error}); free host memory.")
    return host


class BlockStreamer:
    """Own the slot buffers, pinned copies and CUDA events of streamed blocks."""

    def __init__(self, blocks, streamed, *, device, slots=SLOTS):
        self.order = list(streamed)
        self.position = {index: k for k, index in enumerate(self.order)}
        self.slots = min(slots, len(self.order))
        if self.order and len(self.order) % self.slots:
            raise ValueError("Streamed block count must be a multiple of the slots.")
        self.device = torch.device(device)
        self.host: dict[int, torch.Tensor] = {}
        self.arena: torch.Tensor | None = None
        self.buffers: list[torch.Tensor] = []
        self.layout = None
        self.stream = torch.cuda.Stream(self.device) if self.order else None
        self.ready = {i: torch.cuda.Event() for i in self.order}
        self.done = {i: torch.cuda.Event() for i in self.order}
        self.used = set()
        self.blocks = blocks

    def slot_of(self, index: int) -> int:
        return self.position[index] % self.slots

    def offload(self, index: int) -> None:
        """Move one block's forward weights to pinned host memory and slot views."""
        params = forward_parameters(self.blocks[index])
        layout = [(name, p.shape, p.dtype) for name, p in params]
        if len({dtype for _, _, dtype in layout}) != 1:
            raise ValueError("Streamed blocks require a single parameter dtype.")
        if self.layout is None:
            self.layout = layout
            _, numel = _offsets(params)
            dtype = layout[0][2]
            self.buffers = [
                torch.empty(numel, dtype=dtype, device=self.device)
                for _ in range(self.slots)
            ]
            self.arena = page_locked(len(self.order) * numel, dtype)
        elif layout != self.layout:
            raise ValueError("Streamed blocks must share one parameter layout.")
        numel = self.buffers[0].numel()
        k = self.position[index]
        host = self.arena[k * numel : (k + 1) * numel]
        slot = self.buffers[self.slot_of(index)]
        offsets, _ = _offsets(params)
        for (_, p), offset in zip(params, offsets):
            n = p.numel()
            host[offset : offset + n].copy_(p.data.reshape(-1))
            p.data = slot[offset : offset + n].view(p.shape)
        self.host[index] = host

    def prefetch_initial(self) -> None:
        for index in self.order[: self.slots]:
            self._enqueue(index)

    def _enqueue(self, index: int) -> None:
        slot = self.slot_of(index)
        with torch.cuda.stream(self.stream):
            # Wait for the previous user of this slot before overwriting it.
            previous = self.order[(self.position[index] - self.slots) % len(self.order)]
            if previous in self.used:
                self.stream.wait_event(self.done[previous])
            self.buffers[slot].copy_(self.host[index], non_blocking=True)
            self.ready[index].record(self.stream)

    def before(self, index: int) -> None:
        if index in self.position:
            torch.cuda.current_stream(self.device).wait_event(self.ready[index])

    def after(self, index: int) -> None:
        if index not in self.position:
            return
        self.done[index].record(torch.cuda.current_stream(self.device))
        self.used.add(index)
        if len(self.order) <= self.slots:
            return  # Every streamed block owns a slot; its weights stay put.
        upcoming = self.order[(self.position[index] + self.slots) % len(self.order)]
        self._enqueue(upcoming)

    @property
    def host_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.host.values())

    @property
    def slot_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers)


def checkpoint_sizes(path) -> tuple[int, int, int]:
    """Return (per-block forward bytes, non-block bytes, block count) from headers.

    Sizes assume BF16 residency; cross-attention K/V projections are excluded
    because they are dropped after their output is precomputed.
    """
    from safetensors import safe_open

    blocks: dict[int, int] = {}
    other = 0
    with safe_open(str(path), framework="pt") as stream:
        for key in stream.keys():
            shape = stream.get_slice(key).get_shape()
            size = 2
            for d in shape:
                size *= d
            if key.startswith("blocks."):
                if ".attn2.to_k." in key or ".attn2.to_v." in key:
                    continue
                index = int(key.split(".")[1])
                blocks[index] = blocks.get(index, 0) + size
            else:
                other += size
    if not blocks or len(set(blocks.values())) != 1:
        raise ValueError("Checkpoint blocks must share one parameter layout.")
    return next(iter(blocks.values())), other, len(blocks)


def kv_cache_bytes(config: dict, tokens_per_latent: int) -> int:
    """Upper bound of the routed self-attention KV cache for one video."""
    from .cache import ACTIONS

    router = config["static_kv_router"]
    anchors = router["anchor"]["capacity_latents"]
    latents = 0
    for action in router["layer_actions"]:
        recent, use_anchor = ACTIONS[action]
        latents += recent + (anchors if use_anchor else 0)
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    return latents * tokens_per_latent * dim * 2 * 2


def plan_gpu_blocks(
    model_dir, *, device, height, width, reserve_bytes, prompt_tokens=512, slots=SLOTS
) -> tuple[int, dict]:
    """Choose how many DiT blocks stay resident given currently free VRAM."""
    import json
    from pathlib import Path

    from .constants import SPATIAL_TOKEN_STRIDE

    root = Path(model_dir)
    config = json.loads((root / "model_config.json").read_text())
    block, other, count = checkpoint_sizes(root / "transformer.safetensors")
    tokens = (height // SPATIAL_TOKEN_STRIDE) * (width // SPATIAL_TOKEN_STRIDE)
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    kv = kv_cache_bytes(config, tokens)
    cross = count * prompt_tokens * dim * 2 * 2
    free = torch.cuda.mem_get_info(device)[0]
    budget = free - other - kv - cross - reserve_bytes
    if budget >= count * block:
        resident = count
    else:
        resident = max(0, min(count - 1, (budget - slots * block) // block))
        if budget - slots * block < 0:
            raise RuntimeError(
                f"Not enough free VRAM: {free / 2**30:.2f} GiB free, need at least "
                f"{(other + kv + cross + reserve_bytes + slots * block) / 2**30:.2f} GiB."
            )
    info = {
        "free_gib": free / 2**30,
        "block_gib": block / 2**30,
        "non_block_gib": other / 2**30,
        "kv_cache_gib": kv / 2**30,
        "reserve_gib": reserve_bytes / 2**30,
        "resident_blocks": int(resident),
        "streamed_blocks": count - int(resident),
    }
    return int(resident), info
