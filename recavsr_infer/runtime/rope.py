"""Bounded fixed-end temporal RoPE; spatial positions keep the original grid."""

from __future__ import annotations

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from .constants import ROPE_WINDOW_LATENTS


def rotary_table(dim, size, *, device):
    c, s = get_1d_rotary_pos_embed(
        dim,
        size,
        10000.0,
        use_real=True,
        repeat_interleave_real=True,
        freqs_dtype=torch.float64,
    )
    return c.to(device), s.to(device)


def apply_rotary(x, cos, sin):
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[..., 0::2], sin[..., 0::2]
    return (
        torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
        .flatten(-2)
        .to(x.dtype)
    )


def apply_temporal(x, positions, cos, sin, *, span):
    """x has complete planes. cos/sin contain only temporal pair frequencies."""
    tdim = cos.shape[-1] * 2
    planes = positions.numel()
    framed = x.reshape(x.shape[0], planes, span, x.shape[2], x.shape[3])
    c = cos[positions.long()].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    s = sin[positions.long()].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    even = framed[..., :tdim:2].float()
    odd = framed[..., 1:tdim:2].float()
    temporal = (
        torch.stack((even * c - odd * s, even * s + odd * c), dim=-1)
        .flatten(-2)
        .to(x.dtype)
    )
    return torch.cat((temporal, framed[..., tdim:]), dim=-1).reshape_as(x)


class RollingRoPE:
    def __init__(
        self, head_dim, height, width, *, device, training_frames=ROPE_WINDOW_LATENTS
    ):
        if min(head_dim, height, width, training_frames) <= 0:
            raise ValueError("RoPE dimensions must be positive.")
        self.height, self.width, self.training_frames = height, width, training_frames
        self.spatial_dim = 2 * (head_dim // 6)
        self.temporal_dim = head_dim - 2 * self.spatial_dim
        ct, st = rotary_table(self.temporal_dim, training_frames, device=device)
        self.temporal_cos, self.temporal_sin = (
            ct[:, ::2].contiguous(),
            st[:, ::2].contiguous(),
        )
        ch, sh = rotary_table(self.spatial_dim, height, device=device)
        cw, sw = rotary_table(self.spatial_dim, width, device=device)
        shape = (height, width, self.temporal_dim)
        self.spatial_cos = torch.cat(
            (
                torch.ones(shape, device=device),
                ch[:, None].expand(-1, width, -1),
                cw[None].expand(height, -1, -1),
            ),
            -1,
        ).reshape(1, 1, height * width, 1, head_dim)
        self.spatial_sin = torch.cat(
            (
                torch.zeros(shape, device=device),
                sh[:, None].expand(-1, width, -1),
                sw[None].expand(height, -1, -1),
            ),
            -1,
        ).reshape(1, 1, height * width, 1, head_dim)
        self._query_positions = {}

    def positions(self, start: int, frames: int) -> tuple[int, torch.Tensor]:
        """Map absolute latent IDs into the checkpoint's bounded temporal interval."""
        if start < 0 or frames < 1:
            raise ValueError(
                "Latent start must be nonnegative and frame count positive."
            )
        origin = max(0, start + frames - self.training_frames)
        values = tuple(range(start - origin, start - origin + frames))
        if min(values) < 0 or max(values) >= self.training_frames:
            raise ValueError("A block is larger than the rolling interval.")
        if values not in self._query_positions:
            self._query_positions[values] = torch.tensor(
                values,
                device=self.temporal_cos.device,
                dtype=torch.int32,
            )
        return origin, self._query_positions[values]

    def spatial(self, x):
        framed = x.reshape(
            x.shape[0], -1, self.height * self.width, x.shape[2], x.shape[3]
        )
        return apply_rotary(framed, self.spatial_cos, self.spatial_sin).reshape_as(x)
