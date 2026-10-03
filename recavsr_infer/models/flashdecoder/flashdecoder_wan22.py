"""LR-conditioned streaming FlashDecoder-S for Wan2.2 latents.

Inference-only adaptation of this project's trained FlashDecoder implementation.
Inputs are raw 48-channel latents and aligned target-resolution LQ RGB in [-1,1].
LR features are injected before the backbone and temporal refinement; the RGB
head predicts absolute pixels. Wan normalization and output clamping belong to
the runtime adapter, not this model.

Keep the returned state across calls to decode_step: the first latent emits one
RGB frame and later latents emit four. Each latent and its four refinement
frames form a bidirectional block, with a bounded two-block temporal window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn


def _validate_spatial_window(window):
    if (
        window is None
        or len(window) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 2
            for v in window
        )
    ):
        raise ValueError("spatial_window_size requires two positive even integers.")
    return tuple(window)


def _local_attention(
    query,
    key,
    value,
    *,
    height,
    width,
    window,
    group_size,
    temporal_window,
):
    """Streaming local attention; queries are the last complete group of KV."""
    window = _validate_spatial_window(window)
    if min(height, width, group_size, temporal_window) < 1:
        raise ValueError("Local attention dimensions must be positive.")
    n = height * width
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Expected 4D Q and matching 4D K/V.")
    qf, nf = query.shape[2] // n, key.shape[2] // n
    if (
        query.shape[2] % n
        or key.shape[2] % n
        or qf < 1
        or nf < qf
        or qf % group_size
        or nf % group_size
        or query.shape[0] != key.shape[0]
        or query.shape[-1] != key.shape[-1]
        or query.shape[1] % key.shape[1]
    ):
        raise ValueError("Invalid local grid, GQA heads or temporal group alignment.")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q/K/V must share a device.")
    # RMSNorm with FP32 master parameters can promote K/V under BF16 autocast.
    # Preserve the trained SDPA/autocast precision boundary.
    key, value = key.to(query.dtype), value.to(query.dtype)
    supported = (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] == 64
    )
    if supported:
        from ._local_attention_triton import triton_local_attention

        return triton_local_attention(
            query,
            key,
            value,
            height=height,
            width=width,
            window=window,
            group_size=group_size,
            temporal_window=temporal_window,
        )
    # CPU/FP32 diagnostic path; chunk queries to bound the temporary mask.
    k = torch.arange(nf * n, device=query.device)
    kg, ky, kx = k // (group_size * n), (k % n) // width, k % width
    result = []
    wh, ww = window
    for start in range(0, qf * n, 64):
        q = torch.arange(start, min(start + 64, qf * n), device=query.device)
        qg = (q // n + nf - qf) // group_size
        dy = ky[None, :] - ((q % n) // width)[:, None]
        dx = kx[None, :] - (q % width)[:, None]
        mask = (
            (kg[None, :] <= qg[:, None])
            & (kg[None, :] > qg[:, None] - temporal_window)
            & (dy >= -wh // 2)
            & (dy < wh // 2)
            & (dx >= -ww // 2)
            & (dx < ww // 2)
        )
        result.append(
            F.scaled_dot_product_attention(
                query[:, :, start : start + 64],
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                enable_gqa=True,
            )
        )
    return torch.cat(result, dim=2)


FlashDecoderVariant = Literal["S"]


@dataclass(frozen=True)
class _KVCache:
    """Unrotated, normalized keys and values for one Transformer layer."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class FlashDecoderState:
    """Explicit streaming state returned by :meth:`FlashDecoder.decode_step`.

    The container is immutable; a fresh state is required for each video.
    """

    backbone: tuple[_KVCache | None, ...]
    refinement: tuple[_KVCache | None, ...]
    frame_index: int = 0
    batch_size: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None
    spatial_window_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class _VariantSpec:
    depth: int
    dim: int
    num_heads: int
    num_kv_groups: int


class _RotaryEmbedding3D(nn.Module):
    """Relative 3D rotary embeddings for temporal and spatial coordinates."""

    def __init__(
        self,
        *,
        temporal_dim: int = 16,
        height_dim: int = 24,
        width_dim: int = 24,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        dimensions = (temporal_dim, height_dim, width_dim)
        if any(dimension <= 0 or dimension % 2 != 0 for dimension in dimensions):
            raise ValueError(
                f"3D-RoPE dimensions must be positive and even: {dimensions}."
            )
        if theta <= 0:
            raise ValueError(f"RoPE theta must be positive, got {theta}.")

        self.temporal_dim = temporal_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.head_dim = sum(dimensions)

        self.register_buffer(
            "temporal_inv_freq",
            self._build_inv_freq(temporal_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "height_inv_freq",
            self._build_inv_freq(height_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "width_inv_freq",
            self._build_inv_freq(width_dim, theta),
            persistent=False,
        )

    @staticmethod
    def _build_inv_freq(dimension: int, theta: float) -> torch.Tensor:
        indices = torch.arange(0, dimension, 2, dtype=torch.float32)
        return 1.0 / (theta ** (indices / dimension))

    def frequencies(
        self,
        temporal_positions: torch.Tensor,
        height_positions: torch.Tensor,
        width_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real-valued cos/sin tables with shape ``[tokens, head_dim]``."""
        cosines = []
        sines = []
        for positions, inv_freq in (
            (temporal_positions, self.temporal_inv_freq),
            (height_positions, self.height_inv_freq),
            (width_positions, self.width_inv_freq),
        ):
            angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
            cosines.append(torch.cos(angles).repeat_interleave(2, dim=-1))
            sines.append(torch.sin(angles).repeat_interleave(2, dim=-1))
        return (
            torch.cat(cosines, dim=-1).to(dtype=dtype),
            torch.cat(sines, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def apply(
        tensor: torch.Tensor,
        cosines: torch.Tensor,
        sines: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to ``[B, heads, tokens, head_dim]`` tensors."""
        original_dtype = tensor.dtype
        tensor = tensor.to(
            torch.float64 if original_dtype == torch.float64 else torch.float32
        )
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.empty_like(tensor)
        cosines = cosines.unsqueeze(0).unsqueeze(0)
        sines = sines.unsqueeze(0).unsqueeze(0)
        rotated[..., 0::2] = even * cosines[..., 0::2] - odd * sines[..., 0::2]
        rotated[..., 1::2] = even * sines[..., 1::2] + odd * cosines[..., 1::2]
        return rotated.to(original_dtype)


class _GroupedQueryAttention(nn.Module):
    """Grouped-query self-attention with an explicit unrotated KV cache."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = dim // num_heads
        self.kv_dim = num_kv_groups * self.head_dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, self.kv_dim)
        self.value = nn.Linear(dim, self.kv_dim)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.value_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: _KVCache | None,
        query_rope: tuple[torch.Tensor, torch.Tensor],
        key_rope: tuple[torch.Tensor, torch.Tensor],
        local_spec: dict,
    ) -> tuple[torch.Tensor, _KVCache]:
        batch_size, num_tokens, _ = hidden_states.shape

        query = self.query(hidden_states)
        query = query.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key = self.key(hidden_states)
        key = key.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        key = self.key_norm(key)

        value = self.value(hidden_states)
        value = value.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        value = self.value_norm(value)

        if cache is not None:
            key = torch.cat((cache.key, key), dim=2)
            value = torch.cat((cache.value, value), dim=2)

        raw_cache = _KVCache(key=key, value=value)
        query = _RotaryEmbedding3D.apply(query, *query_rope)
        rotated_key = _RotaryEmbedding3D.apply(key, *key_rope)

        attended = _local_attention(query, rotated_key, value, **local_spec)
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.dim)
        return self.output(attended), raw_cache


class _SwiGLU(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class _StreamingTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        hidden_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attention = _GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            norm_eps=norm_eps,
        )
        self.feed_forward_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.feed_forward = _SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: _KVCache | None = None,
        query_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        local_spec: dict,
    ):
        attended, cache = self.attention(
            self.attention_norm(hidden_states),
            cache=cache,
            query_rope=query_rope,
            key_rope=key_rope,
            local_spec=local_spec,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states, cache


class _StreamingTransformer(nn.Module):
    """Transformer stack whose cache is bounded in units of video frames."""

    def __init__(
        self,
        *,
        depth: int,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        hidden_dim: int,
        max_frames: int,
        rope: _RotaryEmbedding3D,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.rope = rope
        self.blocks = nn.ModuleList(
            [
                _StreamingTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_groups=num_kv_groups,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        caches: tuple[_KVCache | None, ...],
        current_frames: int,
        height: int,
        width: int,
        spatial_window_size,
    ) -> tuple[torch.Tensor, tuple[_KVCache, ...]]:
        if len(caches) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} cache entries, got {len(caches)}."
            )
        if current_frames > self.max_frames:
            raise ValueError(
                f"Current chunk has {current_frames} frames, but the cache window "
                f"only supports {self.max_frames}."
            )

        tokens_per_frame = height * width
        current_tokens = current_frames * tokens_per_frame
        if hidden_states.shape[1] != current_tokens:
            raise ValueError(
                f"Expected {current_tokens} tokens for {current_frames} frames at "
                f"{height}x{width}, got {hidden_states.shape[1]}."
            )

        keep_tokens = (self.max_frames - current_frames) * tokens_per_frame
        trimmed_caches = tuple(self._trim_cache(cache, keep_tokens) for cache in caches)
        previous_tokens = self._cache_length(trimmed_caches)
        if previous_tokens % tokens_per_frame != 0:
            raise ValueError(
                f"Cache length {previous_tokens} is not divisible by the "
                f"{tokens_per_frame} tokens in each frame."
            )
        previous_frames = previous_tokens // tokens_per_frame

        query_rope, key_rope = self._build_rope(
            previous_frames=previous_frames,
            current_frames=current_frames,
            height=height,
            width=width,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        local_spec = dict(
            height=height,
            width=width,
            window=spatial_window_size,
            group_size=current_frames,
            temporal_window=self.max_frames // current_frames,
        )
        next_caches = []
        for block, cache in zip(self.blocks, trimmed_caches, strict=True):
            hidden_states, next_cache = block(
                hidden_states,
                cache=cache,
                query_rope=query_rope,
                key_rope=key_rope,
                local_spec=local_spec,
            )
            next_caches.append(next_cache)
        return hidden_states, tuple(next_caches)

    @staticmethod
    def _trim_cache(cache: _KVCache | None, keep_tokens: int) -> _KVCache | None:
        if cache is None or keep_tokens == 0:
            return None
        if cache.key.shape[2] <= keep_tokens:
            return cache
        return _KVCache(
            key=cache.key[:, :, -keep_tokens:, :],
            value=cache.value[:, :, -keep_tokens:, :],
        )

    @staticmethod
    def _cache_length(caches: tuple[_KVCache | None, ...]) -> int:
        lengths = {cache.key.shape[2] for cache in caches if cache is not None}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise ValueError(
                f"Transformer layer cache lengths do not match: {lengths}."
            )
        return lengths.pop()

    def _build_rope(
        self,
        *,
        previous_frames: int,
        current_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        total_frames = previous_frames + current_frames
        key_t, key_h, key_w = self._token_positions(
            start_frame=0,
            num_frames=total_frames,
            height=height,
            width=width,
            device=device,
        )
        query_t, query_h, query_w = self._token_positions(
            start_frame=previous_frames,
            num_frames=current_frames,
            height=height,
            width=width,
            device=device,
        )
        return (
            self.rope.frequencies(
                query_t,
                query_h,
                query_w,
                dtype=torch.float32,
            ),
            self.rope.frequencies(
                key_t,
                key_h,
                key_w,
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _token_positions(
        *,
        start_frame: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal = torch.arange(
            start_frame,
            start_frame + num_frames,
            device=device,
        )
        rows = torch.arange(height, device=device)
        columns = torch.arange(width, device=device)
        temporal, rows, columns = torch.meshgrid(
            temporal,
            rows,
            columns,
            indexing="ij",
        )
        return temporal.flatten(), rows.flatten(), columns.flatten()


class FlashDecoder(nn.Module):
    """Pure-Transformer streaming decoder for Wan2.2-style video latents.

    Wan2.2 inputs are 48-channel VAE latents at 1/16 pixel resolution, before
    the Wan diffusion Transformer's separate 2x2 spatial patch embedding.

    Args:
        latent_channels: Number of channels in each input latent frame.
        out_channels: Number of output pixel channels.
        dim: Transformer model dimension.
        depth: Number of backbone Transformer blocks.
        num_heads: Number of query heads.
        num_kv_groups: Number of grouped key/value heads.
        refinement_depth: Number of temporal refinement blocks.
        mlp_ratio: Hidden expansion ratio for SwiGLU and the spatial MLP.
        temporal_factor: Number of refined temporal tokens per latent frame.
        spatial_factor: PixelShuffle spatial upsampling factor.
        window_size: Rolling backbone window measured in latent frames.
        rope_theta: Base frequency used by 3D-RoPE.
        norm_eps: Epsilon used by all RMSNorm layers.
    """

    VARIANTS: ClassVar[dict[str, _VariantSpec]] = {
        "S": _VariantSpec(depth=12, dim=512, num_heads=8, num_kv_groups=2),
    }

    def __init__(
        self,
        *,
        latent_channels: int = 48,
        out_channels: int = 3,
        dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        refinement_depth: int = 2,
        mlp_ratio: float = 4.0,
        temporal_factor: int = 4,
        spatial_factor: int = 16,
        window_size: int = 2,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-5,
        spatial_window_size: tuple[int, int] = (24, 24),
    ) -> None:
        super().__init__()
        dimensions = (
            latent_channels,
            out_channels,
            dim,
            depth,
            num_heads,
            num_kv_groups,
            refinement_depth,
            temporal_factor,
            spatial_factor,
            window_size,
        )
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in dimensions
        ):
            raise ValueError(
                "Model dimensions, depths and window_size must be positive integers."
            )
        if dim % num_heads or num_heads % num_kv_groups:
            raise ValueError(
                "dim must divide into heads; query heads must divide into KV groups."
            )
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must yield a positive hidden dimension.")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite.")
        self.spatial_window_size = _validate_spatial_window(spatial_window_size)
        hidden_dim = int(dim * mlp_ratio)
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.refinement_depth = refinement_depth
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.window_size = window_size
        if (temporal_factor, spatial_factor, out_channels) != (
            4,
            16,
            3,
        ):
            raise ValueError("LR conditioning requires temporal=4, spatial=16, RGB=3.")
        rope = _RotaryEmbedding3D(theta=rope_theta)
        head_dim = dim // num_heads
        if head_dim != rope.head_dim:
            raise ValueError(
                f"FlashDecoder requires a {rope.head_dim}-dimensional attention "
                f"head for its 16/24/24 3D-RoPE split, got {head_dim}."
            )

        self.latent_projection = nn.Linear(latent_channels, dim)
        self.backbone = _StreamingTransformer(
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            hidden_dim=hidden_dim,
            max_frames=window_size,
            rope=rope,
            norm_eps=norm_eps,
        )
        self.temporal_projection = nn.Linear(dim, temporal_factor * dim)
        self.temporal_refinement = _StreamingTransformer(
            depth=refinement_depth,
            dim=dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            hidden_dim=hidden_dim,
            max_frames=temporal_factor * window_size,
            rope=rope,
            norm_eps=norm_eps,
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels * spatial_factor * spatial_factor),
        )
        # Preserve checkpoint names and parameter construction order.
        self.lr_stem = nn.Linear(3 * 16 * 16, 48)
        self.lr_early_projection = nn.Linear(4 * 48, dim, bias=False)
        self.lr_late_projection = nn.Linear(48, dim, bias=False)

    def _validate_lr(self, latents, lr_up, *, pixel_frames):
        if lr_up is None:
            raise ValueError("Aligned target-resolution lr_up is required.")
        expected = (
            latents.shape[0],
            3,
            pixel_frames,
            latents.shape[-2] * 16,
            latents.shape[-1] * 16,
        )
        if not isinstance(lr_up, torch.Tensor) or tuple(lr_up.shape) != expected:
            raise ValueError(f"lr_up must have target-resolution shape {expected}.")
        if not lr_up.is_floating_point() or lr_up.device != latents.device:
            raise ValueError(
                "lr_up must be floating point on the same device as latents."
            )

    def _encode_lr(self, latents, lr_up, *, first_chunk):
        """Shared framewise stem; return grouped early and aligned late tokens."""
        batch, _, frames, height, width = lr_up.shape
        pixels = lr_up.transpose(1, 2).reshape(batch * frames, 3, height, width)
        packed = F.pixel_unshuffle(pixels, 16).permute(0, 2, 3, 1)
        features = self.lr_stem(packed.to(dtype=latents.dtype))
        height, width = height // 16, width // 16
        features = features.reshape(batch, frames, height, width, 48)
        if first_chunk:
            # Repeat only the actual first frame. Never repad each later chunk.
            features = torch.cat(
                (features[:, :1].expand(-1, 3, -1, -1, -1), features), dim=1
            )
        groups = features.shape[1] // 4
        grouped = (
            features.reshape(batch, groups, 4, height, width, 48)
            .permute(0, 1, 3, 4, 2, 5)
            .reshape(batch, groups * height * width, 192)
        )
        early = self.lr_early_projection(grouped)
        late = self.lr_late_projection(features.reshape(batch, -1, 48))
        return early, late

    @classmethod
    def from_variant(
        cls,
        variant: FlashDecoderVariant = "S",
        **overrides: float,
    ) -> FlashDecoder:
        """Build the released S architecture (12 backbone + 2 refinement blocks)."""
        normalized_variant = variant.upper()
        try:
            spec = cls.VARIANTS[normalized_variant]
        except KeyError as error:
            supported = ", ".join(cls.VARIANTS)
            raise ValueError(
                f"Unknown FlashDecoder variant {variant!r}; expected one of {supported}."
            ) from error

        config: dict[str, int | float] = {
            "depth": spec.depth,
            "dim": spec.dim,
            "num_heads": spec.num_heads,
            "num_kv_groups": spec.num_kv_groups,
        }
        config.update(overrides)
        return cls(**config)

    def init_state(self) -> FlashDecoderState:
        """Create an empty explicit streaming state."""
        return FlashDecoderState(
            backbone=(None,) * self.depth,
            refinement=(None,) * self.refinement_depth,
            spatial_window_size=self.spatial_window_size,
        )

    def decode_step(
        self,
        latent_frame: torch.Tensor,
        state: FlashDecoderState,
        *,
        lr_up: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FlashDecoderState]:
        """Decode one latent frame and return its RGB chunk plus updated state.

        The first latent frame produces one RGB frame. Every subsequent latent
        frame produces ``temporal_factor`` RGB frames.
        """
        if latent_frame.ndim != 4:
            raise ValueError(
                "FlashDecoder.decode_step expects [B,C,H,W], "
                f"got {tuple(latent_frame.shape)}."
            )
        batch_size, channels, height, width = latent_frame.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}."
            )

        if any(size <= 0 for size in latent_frame.shape):
            raise ValueError("Latent frame dimensions cannot be empty.")
        if not latent_frame.is_floating_point():
            raise ValueError("Latents must be floating point.")
        self._validate_state(
            state,
            batch_size=batch_size,
            height=height,
            width=width,
        )

        self._validate_lr(
            latent_frame,
            lr_up,
            pixel_frames=1 if state.frame_index == 0 else 4,
        )
        lr_features = self._encode_lr(
            latent_frame, lr_up, first_chunk=state.frame_index == 0
        )
        hidden_states = latent_frame.permute(0, 2, 3, 1)
        hidden_states = self.latent_projection(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height * width, self.dim)
        hidden_states = hidden_states + lr_features[0]
        hidden_states, backbone_cache = self.backbone(
            hidden_states,
            caches=state.backbone,
            current_frames=1,
            height=height,
            width=width,
            spatial_window_size=self.spatial_window_size,
        )

        hidden_states = self.temporal_projection(hidden_states)
        hidden_states = hidden_states.view(
            batch_size,
            height,
            width,
            self.temporal_factor,
            self.dim,
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2, 4)
        hidden_states = hidden_states.reshape(
            batch_size,
            self.temporal_factor * height * width,
            self.dim,
        )
        hidden_states = hidden_states + lr_features[1]
        hidden_states, refinement_cache = self.temporal_refinement(
            hidden_states,
            caches=state.refinement,
            current_frames=self.temporal_factor,
            height=height,
            width=width,
            spatial_window_size=self.spatial_window_size,
        )

        pixels = self.spatial_projection(hidden_states)
        pixels = pixels.view(
            batch_size,
            self.temporal_factor,
            height,
            width,
            self.out_channels * self.spatial_factor * self.spatial_factor,
        )
        pixels = pixels.permute(0, 1, 4, 2, 3).reshape(
            batch_size * self.temporal_factor,
            self.out_channels * self.spatial_factor * self.spatial_factor,
            height,
            width,
        )
        pixels = F.pixel_shuffle(pixels, self.spatial_factor)
        pixels = pixels.view(
            batch_size,
            self.temporal_factor,
            self.out_channels,
            height * self.spatial_factor,
            width * self.spatial_factor,
        ).permute(0, 2, 1, 3, 4)

        if state.frame_index == 0:
            pixels = pixels[:, :, self.temporal_factor - 1 :, :, :]

        next_state = FlashDecoderState(
            backbone=backbone_cache,
            refinement=refinement_cache,
            frame_index=state.frame_index + 1,
            batch_size=batch_size,
            latent_height=height,
            latent_width=width,
            spatial_window_size=self.spatial_window_size,
        )
        return pixels, next_state

    def _validate_state(
        self,
        state: FlashDecoderState,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        if (
            state.frame_index < 0
            or len(state.backbone) != self.depth
            or len(state.refinement) != self.refinement_depth
        ):
            raise ValueError("Invalid FlashDecoderState index or layer count.")
        if state.spatial_window_size != self.spatial_window_size:
            raise ValueError(
                "Spatial attention changed; start a new FlashDecoderState."
            )
        caches = state.backbone + state.refinement
        if state.frame_index == 0:
            if any(cache is not None for cache in caches):
                raise ValueError("A fresh stream cannot contain history.")
            return
        if any(cache is None for cache in caches):
            raise ValueError("A nonempty stream requires a cache for every layer.")
        expected = (state.batch_size, state.latent_height, state.latent_width)
        actual = (batch_size, height, width)
        if expected != actual:
            raise ValueError(
                f"Streaming shape changed from {expected} to {actual}; "
                "start a new FlashDecoderState for a different stream."
            )


__all__ = [
    "FlashDecoder",
    "FlashDecoderState",
    "FlashDecoderVariant",
]
