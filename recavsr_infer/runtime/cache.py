"""Fixed-storage KV retention; logical recent/anchor choices match ReCaVSR."""

from __future__ import annotations

from functools import lru_cache

import torch

from .constants import ROPE_WINDOW_LATENTS

ACTIONS = {
    "none": (0, False),
    "recent_1": (1, False),
    "recent_2": (2, False),
    "recent_4": (4, False),
    "anchor": (0, True),
    "recent_2_anchor": (2, True),
    "recent_4_anchor": (4, True),
}


@lru_cache(maxsize=256)
def _metadata(device: str, slots: tuple, positions: tuple):
    # Int32 is required by the Triton slot loop; -1 means an unused physical slot.
    return (
        torch.tensor(slots, dtype=torch.int32, device=device),
        torch.tensor(positions, dtype=torch.int32, device=device),
    )


def retained_frames(indices, action, *, initial=5, stride=6, anchors=2):
    """Select a chronological union of recent frames and periodic anchors."""
    recent, use_anchor = ACTIONS[action]
    keep = set(indices[-recent:]) if recent else set()
    if use_anchor:
        keep.update(
            [i for i in indices if i >= initial and (i - initial) % stride == 0][
                -anchors:
            ]
        )
    return tuple(i for i in indices if i in keep)


class FixedKVCache:
    """Bounded KV storage indexed by absolute latent frame, not RGB frame.

    Retention selects logical frame IDs; physical slots remain stable until reused.
    No past activations are recomputed when the rolling RoPE origin advances.
    """

    def __init__(self, action, *, initial=5, stride=6, anchors=2):
        if action not in ACTIONS or stride <= 0 or anchors <= 0:
            raise ValueError("Invalid static KV router.")
        self.action, self.initial, self.stride, self.anchors = (
            action,
            initial,
            stride,
            anchors,
        )
        recent, use_anchor = ACTIONS[action]
        self.capacity = recent + (anchors if use_anchor else 0)
        self.key = self.value = None
        self.frame_to_slot: dict[int, int] = {}
        self.next_latent_frame = 0
        self.signature = None

    def history(self, current, *, origin, train_frames=ROPE_WINDOW_LATENTS):
        """Expose physical buffers, slot IDs and positions relative to the RoPE origin."""
        batch, _, heads, dim = current.shape
        if self.key is None or not self.frame_to_slot:
            empty = current.new_empty((batch, 0, heads, dim))
            slots, positions = _metadata(str(current.device), (), ())
            return empty, empty, slots, positions
        ids = tuple(sorted(self.frame_to_slot))
        mapped = tuple(i - origin for i in ids)
        if min(mapped) < 0 or max(mapped) >= train_frames:
            raise ValueError("Selected KV exceeds the rolling RoPE training interval.")
        slots = tuple(self.frame_to_slot[i] for i in ids) + (-1,) * (
            self.capacity - len(ids)
        )
        positions = mapped + (0,) * (self.capacity - len(ids))
        si, pi = _metadata(str(current.device), slots, positions)
        return self.key, self.value, si, pi

    def commit(self, key, value, *, frames, start):
        """Copy only newly retained KV planes after a successful attention block."""
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("KV tensors must have shape [B,tokens,heads,head_dim].")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("Key and value must share dtype and device.")
        if start != self.next_latent_frame or frames <= 0 or key.shape != value.shape:
            raise ValueError("KV commits must be consecutive, shape-matched blocks.")
        if key.shape[1] % frames:
            raise ValueError("KV length must contain complete spatial planes.")
        span = key.shape[1] // frames
        signature = (key.shape[0], span, key.shape[2:], key.dtype, key.device)
        if self.signature is not None and self.signature != signature:
            raise ValueError("KV geometry changed within a video; reset the state.")
        self.signature = signature
        current_ids = tuple(range(start, start + frames))
        candidates = tuple(sorted(self.frame_to_slot)) + current_ids
        selected = retained_frames(
            candidates,
            self.action,
            initial=self.initial,
            stride=self.stride,
            anchors=self.anchors,
        )
        self.next_latent_frame += frames
        if not self.capacity:
            return
        if self.key is None:
            shape = (key.shape[0], self.capacity * span, *key.shape[2:])
            self.key, self.value = key.new_empty(shape), value.new_empty(shape)
        self.frame_to_slot = {
            i: s for i, s in self.frame_to_slot.items() if i in selected
        }
        free = iter(
            sorted(set(range(self.capacity)) - set(self.frame_to_slot.values()))
        )
        for i in selected:
            if i in self.frame_to_slot:
                continue
            slot = next(free)
            offset = i - start
            if offset < 0:
                raise RuntimeError("Retention tried to resurrect an evicted KV frame.")
            self.key[:, slot * span : (slot + 1) * span].copy_(
                key[:, offset * span : (offset + 1) * span]
            )
            self.value[:, slot * span : (slot + 1) * span].copy_(
                value[:, offset * span : (offset + 1) * span]
            )
            self.frame_to_slot[i] = slot

    @property
    def frame_indices(self):
        return tuple(sorted(self.frame_to_slot))
