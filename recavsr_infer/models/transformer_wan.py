# Modified for ReCaVSR: streaming VSR conditioning and loading-only inference.
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models._modeling_parallel import (
    ContextParallelInput,
    ContextParallelOutput,
)
from diffusers.models.attention import (
    AttentionMixin,
    AttentionModuleMixin,
    FeedForward,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_compiled_flex_attention = None


_STATIC_KV_CACHE_ACTIONS: dict[str, tuple[int, bool]] = {
    "none": (0, False),
    "recent_1": (1, False),
    "recent_2": (2, False),
    "recent_4": (4, False),
    "anchor": (0, True),
    "recent_2_anchor": (2, True),
    "recent_4_anchor": (4, True),
}


@dataclass(frozen=True)
class WanStaticKVRouterConfig:
    """Validated training-time static historical-attention schedule."""

    layer_actions: tuple[str, ...]
    anchor_initial_index: int
    anchor_stride_latents: int
    anchor_capacity_latents: int


def _parse_static_kv_router_config(
    config: Mapping[str, object] | None,
    *,
    num_layers: int,
) -> WanStaticKVRouterConfig | None:
    if config is None or not bool(config.get("enabled", True)):
        return None
    unit = str(config.get("unit", "latent_frame"))
    if unit != "latent_frame":
        raise ValueError(
            "static_kv_router.unit must be 'latent_frame', "
            f"got {unit!r}."
        )
    raw_actions = config.get("layer_actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise TypeError("static_kv_router.layer_actions must be a sequence.")
    layer_actions = tuple(str(action) for action in raw_actions)
    if len(layer_actions) != num_layers:
        raise ValueError(
            "static_kv_router.layer_actions must match the transformer depth: "
            f"actions={len(layer_actions)}, layers={num_layers}."
        )
    invalid_actions = sorted(set(layer_actions) - _STATIC_KV_CACHE_ACTIONS.keys())
    if invalid_actions:
        raise ValueError(
            "Unsupported static KV cache actions: "
            f"{invalid_actions}; expected one of {sorted(_STATIC_KV_CACHE_ACTIONS)}."
        )

    anchor = config.get("anchor", {})
    if not isinstance(anchor, Mapping):
        raise TypeError("static_kv_router.anchor must be a mapping.")
    anchor_initial_index = int(anchor.get("initial_index", 5))
    anchor_stride_latents = int(anchor.get("stride_latents", 8))
    anchor_capacity_latents = int(anchor.get("capacity_latents", 2))
    if anchor_initial_index < 0:
        raise ValueError("static KV anchor initial_index must be non-negative.")
    if anchor_stride_latents <= 0:
        raise ValueError("static KV anchor stride_latents must be positive.")
    if anchor_capacity_latents <= 0:
        raise ValueError("static KV anchor capacity_latents must be positive.")

    return WanStaticKVRouterConfig(
        layer_actions=layer_actions,
        anchor_initial_index=anchor_initial_index,
        anchor_stride_latents=anchor_stride_latents,
        anchor_capacity_latents=anchor_capacity_latents,
    )


@dataclass
class WanSelfAttentionKVCache:
    """Bounded per-layer self-attention history for chunk streaming."""

    window_size: int = -1
    action: str | None = None
    anchor_initial_index: int = 0
    anchor_stride_latents: int = 1
    anchor_capacity_latents: int = 1
    key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    tokens_per_frame: int | None = None
    frame_indices: tuple[int, ...] = ()
    next_frame_index: int = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_frames <= 0 or key.shape[1] % num_frames != 0:
            raise ValueError(
                "Streaming KV tokens must divide evenly into latent frames: "
                f"tokens={key.shape[1]}, frames={num_frames}."
            )
        current_tokens = key.shape[1] // num_frames
        if self.tokens_per_frame is None:
            self.tokens_per_frame = current_tokens
        elif self.tokens_per_frame != current_tokens:
            raise ValueError(
                "Streaming KV cache requires a constant token count per frame."
            )

        current_frame_indices = tuple(
            range(self.next_frame_index, self.next_frame_index + num_frames)
        )
        self.next_frame_index += num_frames
        attention_key = key
        attention_value = value
        if self.key_value is not None:
            cached_key, cached_value = self.key_value
            attention_key = torch.cat((cached_key, key), dim=1)
            attention_value = torch.cat((cached_value, value), dim=1)

        candidate_indices = self.frame_indices + current_frame_indices
        retained_indices = self._select_retained_frames(candidate_indices)
        if retained_indices:
            candidate_positions = {
                frame_index: position
                for position, frame_index in enumerate(candidate_indices)
            }
            retained_positions = [
                candidate_positions[frame_index] for frame_index in retained_indices
            ]
            frame_key = attention_key.unflatten(
                1, (len(candidate_indices), current_tokens)
            )
            frame_value = attention_value.unflatten(
                1, (len(candidate_indices), current_tokens)
            )
            self.key_value = (
                frame_key[:, retained_positions].flatten(1, 2).detach(),
                frame_value[:, retained_positions].flatten(1, 2).detach(),
            )
        else:
            self.key_value = None
        self.frame_indices = retained_indices
        return attention_key, attention_value

    def _select_retained_frames(
        self,
        candidate_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not candidate_indices:
            return ()
        if self.action is None:
            if self.window_size == -1:
                return candidate_indices
            return candidate_indices[-self.window_size :]

        recent_window, use_anchor = _STATIC_KV_CACHE_ACTIONS[self.action]
        retained: set[int] = set()
        if recent_window > 0:
            retained.update(candidate_indices[-recent_window:])
        if use_anchor:
            anchors = [
                frame_index
                for frame_index in candidate_indices
                if frame_index >= self.anchor_initial_index
                and (frame_index - self.anchor_initial_index)
                % self.anchor_stride_latents
                == 0
            ]
            retained.update(anchors[-self.anchor_capacity_latents :])
        return tuple(
            frame_index
            for frame_index in candidate_indices
            if frame_index in retained
        )


@dataclass
class WanCrossAttentionKVCache:
    """Projected text keys and values shared by all matching streaming samples."""

    key_value: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class WanConditioningCache:
    """Static one-step conditioning reused across frames and input samples."""

    temb: torch.Tensor
    timestep_proj: torch.Tensor
    encoder_hidden_states: torch.Tensor
    cross_attention_caches: list[WanCrossAttentionKVCache]


@dataclass
class WanStreamingState:
    """Mutable state owned by one streaming timestep and one video batch."""

    block_caches: list[WanSelfAttentionKVCache]
    conditioning_cache: WanConditioningCache | None = None
    lq_projection_state: object | None = None
    frame_index: int = 0


def _get_qkv_projections(
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


class WanAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: BlockMask | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        streaming_num_frames: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cross_key_value is None:
            query, key, value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
            key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
        else:
            if encoder_hidden_states is None or not attn.is_cross_attention:
                raise ValueError(
                    "Cross-attention KV cache requires cross-attention encoder states."
                )
            if rotary_emb is not None or block_mask is not None:
                raise ValueError(
                    "Cross-attention KV cache does not support rotary embeddings or block masks."
                )
            query = attn.to_q(hidden_states)
            key, value = cross_key_value

        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        if streaming_num_frames is not None:
            if encoder_hidden_states is not None:
                raise ValueError(
                    "Streaming KV cache is only supported for self-attention."
                )
            if attention_mask is not None or block_mask is not None:
                raise ValueError(
                    "Streaming self-attention does not accept an attention mask."
                )

            # Only tensor snapshots cross the AC/compile boundary. The caller
            # commits current K/V after the block returns, never during replay.
            current_key, current_value = key, value
            if past_key_value is not None:
                key = torch.cat((past_key_value[0], key), dim=1)
                value = torch.cat((past_key_value[1], value), dim=1)

        if block_mask is not None:
            hidden_states = _run_flex_attention(
                query, key, value, attention_mask, block_mask
            )
        else:
            if (
                attention_mask is not None
                and attention_mask.ndim == 2
                and attention_mask.shape[0] == query.shape[0]
                and attention_mask.shape[1] == key.shape[1]
            ):
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

            hidden_states = F.scaled_dot_product_attention(
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if streaming_num_frames is not None:
            return hidden_states, current_key, current_value
        return hidden_states


def _run_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    block_mask: BlockMask | None,
) -> torch.Tensor:
    if attention_mask is not None:
        raise ValueError(
            "flex_attention backend does not support attention_mask; use block_mask instead."
        )

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    original_query_length = query.shape[2]

    if block_mask is not None:
        query, key, value = _pad_qkv_for_block_mask(query, key, value, block_mask)

    if not query.is_cuda and torch.is_grad_enabled():
        raise RuntimeError(
            "flex_attention backend only supports training on CUDA devices."
        )
    if query.is_cuda:
        attention_fn = _get_cuda_flex_attention()
        hidden_states = attention_fn(
            query=query, key=key, value=value, block_mask=block_mask
        )
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="flex_attention called without torch.compile"
            )
            hidden_states = _flex_attention(
                query=query, key=key, value=value, block_mask=block_mask
            )

    hidden_states = hidden_states[:, :, :original_query_length]
    return hidden_states.transpose(1, 2)


def _get_cuda_flex_attention():
    # When a containing transformer block is compiled, let that outer graph
    # capture FlexAttention and lower it with Inductor. Calling an independently
    # compiled function here would introduce a nested compile boundary. Keep the
    # standalone compiled function as the eager-model fallback.
    if torch.compiler.is_compiling():
        return _flex_attention
    return _get_compiled_flex_attention()


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            _flex_attention,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
    return _compiled_flex_attention


def _pad_qkv_for_block_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_length, key_length = block_mask.seq_lengths
    query = _pad_sequence_dim(query, query_length)
    key = _pad_sequence_dim(key, key_length)
    value = _pad_sequence_dim(value, key_length)
    return query, key, value


def _pad_sequence_dim(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    padding_length = target_length - hidden_states.shape[2]
    if padding_length < 0:
        raise ValueError(
            f"Block mask length {target_length} is shorter than attention sequence length {hidden_states.shape[2]}."
        )
    if padding_length == 0:
        return hidden_states

    padding_shape = list(hidden_states.shape)
    padding_shape[2] = padding_length
    padding = hidden_states.new_zeros(padding_shape)
    return torch.cat([hidden_states, padding], dim=2)


def _validate_attention_config(
    *,
    num_frames_per_block: int,
    causal_prefix_num_latent_frames: int,
    causal_history_num_latent_frames: int,
    local_attn_size: int,
    sink_size: int,
    flex_attention_block_size: int,
) -> None:
    if num_frames_per_block <= 0:
        raise ValueError(
            f"num_frames_per_block must be positive, got {num_frames_per_block}."
        )
    if causal_prefix_num_latent_frames < 0:
        raise ValueError(
            "causal_prefix_num_latent_frames must be non-negative, got "
            f"{causal_prefix_num_latent_frames}."
        )
    if causal_history_num_latent_frames < -1:
        raise ValueError(
            "causal_history_num_latent_frames must be -1 or non-negative, got "
            f"{causal_history_num_latent_frames}."
        )
    if local_attn_size < -1:
        raise ValueError(
            f"local_attn_size must be -1 or non-negative, got {local_attn_size}."
        )
    if sink_size < 0:
        raise ValueError(f"sink_size must be non-negative, got {sink_size}.")
    if flex_attention_block_size <= 0:
        raise ValueError(
            f"flex_attention_block_size must be positive, got {flex_attention_block_size}."
        )


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat(
                [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
            )
            concatenated_bias = torch.cat(
                [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
            )
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )
        else:
            concatenated_weights = torch.cat(
                [self.to_k.weight.data, self.to_v.weight.data]
            )
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def precompute_cross_attention_kv(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_cross_attention:
            raise ValueError("Cross-attention KV precompute requires cross-attention.")
        if self.fused_projections:
            key, value = self.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: BlockMask | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            block_mask,
            **kwargs,
        )


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        del batch_size, num_channels
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        if frame_offset < 0:
            raise ValueError(f"frame_offset must be non-negative, got {frame_offset}.")
        if frame_offset + ppf > self.max_seq_len:
            raise ValueError(
                "Temporal rotary position exceeds rope_max_seq_len: "
                f"offset={frame_offset}, frames={ppf}, max={self.max_seq_len}."
            )

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        temporal_slice = slice(frame_offset, frame_offset + ppf)
        freqs_cos_f = (
            freqs_cos[0][temporal_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        )
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = (
            freqs_sin[0][temporal_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        )
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        block_mask: BlockMask | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        precompute_cross_attention_only: bool = False,
        streaming_num_frames: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if precompute_cross_attention_only:
            key, value = self.attn2.precompute_cross_attention_kv(
                encoder_hidden_states,
            )
            return hidden_states, key, value

        assert temb.ndim == 4
        # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        # batch_size, seq_len, 1, inner_dim
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            None,
            rotary_emb,
            block_mask,
            past_key_value=past_key_value,
            streaming_num_frames=streaming_num_frames,
        )
        if streaming_num_frames is not None:
            attn_output, current_key, current_value = attn_output
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
            cross_key_value=cross_key_value,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        if streaming_num_frames is not None:
            return hidden_states, current_key, current_value
        return hidden_states


class WanTransformer3DModel(
    ModelMixin,
    ConfigMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(
                split_dim=1, expected_dims=3, split_output=False
            ),
        },
        # Reference: https://github.com/huggingface/diffusers/pull/12909
        # We need to disable the splitting of encoder_hidden_states because the image_encoder
        # (Wan 2.1 I2V) consistently generates 257 tokens for image_embed. This causes the shape
        # of encoder_hidden_states—whose token count is always 769 (512 + 257) after concatenation
        # —to be indivisible by the number of devices in the CP.
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(
                split_dim=1, expected_dims=2, split_output=False
            ),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        lq_proj_hidden_dim1: int = 2048,
        lq_proj_hidden_dim2: int = 3072,
        recycle_enabled: bool = False,
        use_block_causal_attention: bool = False,
        num_frames_per_block: int = 1,
        causal_prefix_num_latent_frames: int = 0,
        causal_history_num_latent_frames: int = -1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        flex_attention_block_size: int = 128,
        static_kv_router: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        _validate_attention_config(
            num_frames_per_block=num_frames_per_block,
            causal_prefix_num_latent_frames=causal_prefix_num_latent_frames,
            causal_history_num_latent_frames=causal_history_num_latent_frames,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            flex_attention_block_size=flex_attention_block_size,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self._static_kv_router = _parse_static_kv_router_config(
            static_kv_router,
            num_layers=len(self.blocks),
        )
        if self._static_kv_router is not None:
            if not use_block_causal_attention:
                raise ValueError(
                    "static_kv_router requires use_block_causal_attention=True."
                )
            if causal_history_num_latent_frames != -1:
                raise ValueError(
                    "static_kv_router replaces the uniform causal history; set "
                    "causal_history_num_latent_frames=-1."
                )
            if local_attn_size != -1 or sink_size != 0:
                raise ValueError(
                    "static_kv_router is incompatible with local_attn_size/sink_size; "
                    "set local_attn_size=-1 and sink_size=0."
                )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False
        self._block_causal_mask_cache: dict[tuple, BlockMask] = {}

    def _get_block_causal_attention_masks(
        self,
        *,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
    ) -> tuple[BlockMask | None, ...]:
        # A single frame has no temporal future to mask. Returning None routes
        # self-attention through SDPA and avoids Flex Attention's block-mask
        # construction, while later video batches still build and reuse masks.
        if not self.config.use_block_causal_attention or num_frames == 1:
            return (None,) * len(self.blocks)

        layer_actions: tuple[str | None, ...]
        if self._static_kv_router is None:
            layer_actions = (None,) * len(self.blocks)
        else:
            layer_actions = self._static_kv_router.layer_actions

        return tuple(
            self._get_block_causal_attention_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                static_kv_cache_action=action,
            )
            for action in layer_actions
        )

    def _get_block_causal_attention_mask(
        self,
        *,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        static_kv_cache_action: str | None,
    ) -> BlockMask:
        cache_key = self._block_causal_mask_cache_key(
            device,
            num_frames,
            frame_seqlen,
            static_kv_cache_action=static_kv_cache_action,
        )

        block_mask = self._block_causal_mask_cache.get(cache_key)
        if block_mask is None:
            anchor_initial_index = 0
            anchor_stride_latents = 1
            anchor_capacity_latents = 1
            if self._static_kv_router is not None:
                anchor_initial_index = self._static_kv_router.anchor_initial_index
                anchor_stride_latents = self._static_kv_router.anchor_stride_latents
                anchor_capacity_latents = self._static_kv_router.anchor_capacity_latents
            block_mask = self._prepare_block_causal_attention_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frames_per_block=int(self.config.num_frames_per_block),
                causal_prefix_num_latent_frames=int(
                    self.config.causal_prefix_num_latent_frames
                ),
                causal_history_num_latent_frames=int(
                    self.config.causal_history_num_latent_frames
                ),
                local_attn_size=int(self.config.local_attn_size),
                sink_size=int(self.config.sink_size),
                flex_attention_block_size=int(self.config.flex_attention_block_size),
                static_kv_cache_action=static_kv_cache_action,
                anchor_initial_index=anchor_initial_index,
                anchor_stride_latents=anchor_stride_latents,
                anchor_capacity_latents=anchor_capacity_latents,
            )
            self._block_causal_mask_cache[cache_key] = block_mask
            logger.info(
                "Cached block causal attention mask: frames=%s, frame_seqlen=%s, "
                "frames_per_block=%s, prefix_frames=%s, history_frames=%s, "
                "static_action=%s",
                num_frames,
                frame_seqlen,
                self.config.num_frames_per_block,
                self.config.causal_prefix_num_latent_frames,
                self.config.causal_history_num_latent_frames,
                static_kv_cache_action,
            )
        return block_mask

    def _block_causal_mask_cache_key(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        *,
        static_kv_cache_action: str | None,
    ) -> tuple:
        device = torch.device(device)
        static_router_key = None
        if self._static_kv_router is not None:
            static_router_key = (
                self._static_kv_router.anchor_initial_index,
                self._static_kv_router.anchor_stride_latents,
                self._static_kv_router.anchor_capacity_latents,
            )
        return (
            device.type,
            device.index,
            int(num_frames),
            int(frame_seqlen),
            int(self.config.num_frames_per_block),
            int(self.config.causal_prefix_num_latent_frames),
            int(self.config.causal_history_num_latent_frames),
            int(self.config.local_attn_size),
            int(self.config.sink_size),
            int(self.config.flex_attention_block_size),
            static_kv_cache_action,
            static_router_key,
        )

    @staticmethod
    def _prepare_block_causal_attention_mask(
        *,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        num_frames_per_block: int = 1,
        causal_prefix_num_latent_frames: int = 0,
        causal_history_num_latent_frames: int = -1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        flex_attention_block_size: int = 128,
        static_kv_cache_action: str | None = None,
        anchor_initial_index: int = 0,
        anchor_stride_latents: int = 1,
        anchor_capacity_latents: int = 1,
    ) -> BlockMask:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}.")
        if frame_seqlen <= 0:
            raise ValueError(f"frame_seqlen must be positive, got {frame_seqlen}.")
        _validate_attention_config(
            num_frames_per_block=num_frames_per_block,
            causal_prefix_num_latent_frames=causal_prefix_num_latent_frames,
            causal_history_num_latent_frames=causal_history_num_latent_frames,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            flex_attention_block_size=flex_attention_block_size,
        )
        if (
            static_kv_cache_action is not None
            and static_kv_cache_action not in _STATIC_KV_CACHE_ACTIONS
        ):
            raise ValueError(
                f"Unsupported static KV cache action {static_kv_cache_action!r}."
            )
        if anchor_initial_index < 0:
            raise ValueError("anchor_initial_index must be non-negative.")
        if anchor_stride_latents <= 0:
            raise ValueError("anchor_stride_latents must be positive.")
        if anchor_capacity_latents <= 0:
            raise ValueError("anchor_capacity_latents must be positive.")

        recent_window_value = 0
        use_anchor_value = False
        if static_kv_cache_action is not None:
            recent_window_value, use_anchor_value = _STATIC_KV_CACHE_ACTIONS[
                static_kv_cache_action
            ]
        use_static_router = static_kv_cache_action is not None

        total_length = num_frames * frame_seqlen
        padded_length = (
            math.ceil(total_length / flex_attention_block_size)
            * flex_attention_block_size
        )
        prefix_frames = min(causal_prefix_num_latent_frames, num_frames)
        token_indices = torch.arange(padded_length, device=device, dtype=torch.long)
        # Keep per-action router controls as tensor inputs to mask_mod. Capturing
        # Python ints/bools here makes Dynamo specialize WanTransformerBlock.forward
        # once per action and quickly exhaust its shared recompile cache.
        recent_window = torch.tensor(
            recent_window_value,
            device=device,
            dtype=torch.long,
        )
        use_anchor = torch.tensor(
            use_anchor_value,
            device=device,
            dtype=torch.bool,
        )
        frame_indices = token_indices // frame_seqlen
        is_prefix_frame = frame_indices < prefix_frames
        post_prefix_frames = (frame_indices - prefix_frames).clamp(min=0)
        chunk_starts = prefix_frames + (
            post_prefix_frames // num_frames_per_block
        ) * num_frames_per_block
        chunk_starts = torch.where(
            is_prefix_frame,
            torch.zeros_like(chunk_starts),
            chunk_starts,
        )
        chunk_ends = torch.where(
            is_prefix_frame,
            torch.full_like(chunk_starts, prefix_frames),
            chunk_starts + num_frames_per_block,
        ).clamp(max=num_frames)
        sink_token_length = sink_size * frame_seqlen
        local_window_length = local_attn_size * frame_seqlen

        def block_causal_mask(_batch, _head, query_index, key_index):
            is_valid_query = query_index < total_length
            is_valid_key = key_index < total_length
            same_padding_token = query_index == key_index
            key_frame = key_index // frame_seqlen
            chunk_start = chunk_starts[query_index]
            chunk_end = chunk_ends[query_index]

            is_block_causal = is_valid_query & is_valid_key & (key_frame < chunk_end)

            if use_static_router:
                in_current_chunk = (key_frame >= chunk_start) & (
                    key_frame < chunk_end
                )
                recent_start = torch.clamp(chunk_start - recent_window, min=0)
                in_recent = (
                    (key_frame >= recent_start)
                    & (key_frame < chunk_start)
                    & (recent_window > 0)
                )

                has_anchor = chunk_start > anchor_initial_index
                latest_anchor_step = torch.div(
                    chunk_start - 1 - anchor_initial_index,
                    anchor_stride_latents,
                    rounding_mode="floor",
                )
                latest_anchor = (
                    anchor_initial_index
                    + latest_anchor_step * anchor_stride_latents
                )
                earliest_anchor = latest_anchor - (
                    anchor_capacity_latents - 1
                ) * anchor_stride_latents
                in_anchor = (
                    has_anchor
                    & (key_frame >= earliest_anchor)
                    & (key_frame <= latest_anchor)
                    & (
                        (key_frame - anchor_initial_index)
                        % anchor_stride_latents
                        == 0
                    )
                    & use_anchor
                )
                visible = in_current_chunk | in_recent | in_anchor
                return (is_block_causal & visible) | same_padding_token

            if causal_history_num_latent_frames != -1:
                visible_start = torch.clamp(
                    chunk_start - causal_history_num_latent_frames,
                    min=0,
                )
                in_history = key_frame >= visible_start
                in_sink = key_index < sink_token_length
                return (
                    is_block_causal & (in_history | in_sink)
                ) | same_padding_token

            if local_attn_size == -1:
                return is_block_causal | same_padding_token

            chunk_end_token = chunk_end * frame_seqlen
            in_window = key_index >= (chunk_end_token - local_window_length)
            in_sink = key_index < sink_token_length
            return (
                is_block_causal & (in_window | in_sink | same_padding_token)
            ) | same_padding_token

        return create_block_mask(
            block_causal_mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            BLOCK_SIZE=flex_attention_block_size,
            _compile=False,
            device=device,
        )

    def build_lq_proj(self):
        from .lq_proj import LQ4xProj

        return LQ4xProj(
            in_dim=3,
            out_dim=self.config.num_attention_heads * self.config.attention_head_dim,
            hidden_dim1=self.config.lq_proj_hidden_dim1,
            hidden_dim2=self.config.lq_proj_hidden_dim2,
        )

    def build_sr_proj(self) -> nn.Linear:
        """Build the zero-initialized previous-SR latent projection."""
        inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        projection = nn.Linear(inner_dim, inner_dim)
        nn.init.zeros_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)
        return projection

    def create_streaming_state(
        self,
        *,
        kv_cache_window_size: int = -1,
        conditioning_cache: WanConditioningCache | None = None,
    ) -> WanStreamingState:
        """Create empty layer-wise KV caches for one-step chunk streaming."""
        if not self.config.use_block_causal_attention:
            raise ValueError(
                "Streaming inference requires use_block_causal_attention=True."
            )
        if int(self.config.local_attn_size) != -1 or int(self.config.sink_size) != 0:
            raise ValueError(
                "Streaming inference currently requires local_attn_size=-1 and sink_size=0."
            )
        if kv_cache_window_size == 0 or kv_cache_window_size < -1:
            raise ValueError("kv_cache_window_size must be -1 or a positive integer.")
        if self._static_kv_router is not None and kv_cache_window_size != -1:
            raise ValueError(
                "kv_cache_window_size cannot override model.static_kv_router."
            )
        layer_actions: tuple[str | None, ...]
        if self._static_kv_router is None:
            layer_actions = (None,) * len(self.blocks)
            anchor_initial_index = 0
            anchor_stride_latents = 1
            anchor_capacity_latents = 1
        else:
            layer_actions = self._static_kv_router.layer_actions
            anchor_initial_index = self._static_kv_router.anchor_initial_index
            anchor_stride_latents = self._static_kv_router.anchor_stride_latents
            anchor_capacity_latents = self._static_kv_router.anchor_capacity_latents
        return WanStreamingState(
            block_caches=[
                WanSelfAttentionKVCache(
                    window_size=kv_cache_window_size,
                    action=action,
                    anchor_initial_index=anchor_initial_index,
                    anchor_stride_latents=anchor_stride_latents,
                    anchor_capacity_latents=anchor_capacity_latents,
                )
                for action in layer_actions
            ],
            conditioning_cache=conditioning_cache,
        )

    def _embed_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            flattened_timestep = timestep.flatten()
        else:
            timestep_seq_len = None
            flattened_timestep = timestep
        temb, timestep_proj, projected_encoder_hidden_states = self.condition_embedder(
            flattened_timestep,
            encoder_hidden_states,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        return temb, timestep_proj, projected_encoder_hidden_states

    def _cache_streaming_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> WanConditioningCache:
        if encoder_hidden_states.shape[0] != 1:
            raise ValueError("Streaming conditioning requires prompt batch size 1.")
        if timestep.ndim == 2:
            base_timestep = timestep[:, :1]
        elif timestep.ndim == 1:
            base_timestep = timestep[:, None]
        else:
            raise ValueError(
                "Streaming timestep must have shape [batch] or [batch, tokens]."
            )
        if base_timestep.shape[0] != 1:
            raise ValueError("Streaming conditioning requires timestep batch size 1.")

        temb, timestep_proj, projected_encoder_hidden_states = self.condition_embedder(
            base_timestep.flatten(),
            encoder_hidden_states,
            timestep_seq_len=1,
        )
        cache = WanConditioningCache(
            temb=temb.detach(),
            timestep_proj=timestep_proj.unflatten(2, (6, -1)).detach(),
            encoder_hidden_states=projected_encoder_hidden_states.detach(),
            cross_attention_caches=[WanCrossAttentionKVCache() for _ in self.blocks],
        )
        return cache

    def _expand_streaming_conditioning(
        self,
        *,
        streaming_state: WanStreamingState,
        batch_size: int,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = streaming_state.conditioning_cache
        if cache is None:
            raise RuntimeError(
                "Streaming conditioning was not prepared; call "
                "pipeline.prepare_streaming() before sampling."
            )
        return (
            cache.temb.expand(batch_size, token_count, -1),
            cache.timestep_proj.expand(batch_size, token_count, -1, -1),
            cache.encoder_hidden_states.expand(batch_size, -1, -1),
        )

    def _precompute_streaming(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        streaming_state: WanStreamingState | None,
    ) -> torch.Tensor:
        """Populate streaming conditioning caches through transformer block forwards."""
        if streaming_state is None:
            raise ValueError("Streaming precompute requires a streaming state.")
        batch_size, _, num_frames, _, _ = hidden_states.shape
        if num_frames != 1 or self.config.patch_size[0] != 1:
            raise ValueError(
                "Streaming precompute requires a one-frame latent and temporal patch_size=1."
            )

        cache = self._cache_streaming_conditioning(
            timestep,
            encoder_hidden_states,
        )
        streaming_state.conditioning_cache = cache
        empty_hidden_states = cache.encoder_hidden_states.new_empty(
            (
                batch_size,
                0,
                self.config.num_attention_heads * self.config.attention_head_dim,
            )
        )
        for index, block in enumerate(self.blocks):
            empty_hidden_states, key, value = block(
                empty_hidden_states,
                cache.encoder_hidden_states,
                cache.timestep_proj,
                (self.rope.freqs_cos, self.rope.freqs_sin),
                precompute_cross_attention_only=True,
            )
            cache.cross_attention_caches[index].key_value = key.detach(), value.detach()
        return hidden_states

    def forward(
        self,
        lq_videos: torch.Tensor | None,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        extract_layers: Sequence[int] | None = None,
        use_lq_condition: bool = True,
        recycle_latents: torch.Tensor | None = None,
        streaming_state: WanStreamingState | None = None,
        precompute_streaming: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run full-sequence, one-frame streaming, or streaming precompute."""
        if precompute_streaming:
            return self._precompute_streaming(
                hidden_states,
                timestep,
                encoder_hidden_states,
                streaming_state,
            )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        del num_channels
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if streaming_state is not None:
            if extract_layers is not None:
                raise ValueError(
                    "extract_layers is not supported during streaming inference."
                )
            if p_t != 1:
                raise ValueError(
                    "Streaming inference requires temporal patch_size=1."
                )
            prefix_size = int(self.config.causal_prefix_num_latent_frames)
            block_size = int(self.config.num_frames_per_block)
            if streaming_state.frame_index == 0 and prefix_size > 0:
                if post_patch_num_frames != prefix_size:
                    raise ValueError(
                        "The first streaming call must contain the complete "
                        f"bidirectional prefix: expected={prefix_size}, "
                        f"got={post_patch_num_frames}."
                    )
            elif not 1 <= post_patch_num_frames <= block_size:
                raise ValueError(
                    "A post-prefix streaming call must contain at most one "
                    f"causal block: block_size={block_size}, "
                    f"got={post_patch_num_frames}."
                )
            if len(streaming_state.block_caches) != len(self.blocks):
                raise ValueError(
                    "Streaming state block count does not match the transformer."
                )
            rotary_emb = self.rope(
                hidden_states,
                frame_offset=streaming_state.frame_index,
            )
        else:
            rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if use_lq_condition:
            if streaming_state is None:
                if lq_videos is None:
                    raise ValueError("lq_videos is required when use_lq_condition=True.")
                lq_condition = self.lq_proj(lq_videos)
            else:
                if lq_videos is None:
                    raise ValueError(
                        "Every streaming chunk requires its corresponding LQ frames."
                    )
                if hasattr(self.lq_proj, "forward_streaming"):
                    if streaming_state.lq_projection_state is None:
                        streaming_state.lq_projection_state = (
                            self.lq_proj.create_streaming_state()
                        )
                    lq_condition = self.lq_proj.forward_streaming(
                        lq_videos,
                        streaming_state.lq_projection_state,
                    )
                else:
                    lq_condition = self.lq_proj(lq_videos)
            if lq_condition.shape != hidden_states.shape:
                raise ValueError(
                    "LQ condition tokens must match Wan latent tokens: "
                    f"lq={tuple(lq_condition.shape)}, "
                    f"latent={tuple(hidden_states.shape)}."
                )
            hidden_states = hidden_states + lq_condition
            if self.config.recycle_enabled:
                if recycle_latents is None:
                    raise ValueError(
                        "recycle_latents is required when recycle_enabled=True."
                    )
                recycle_condition = self.patch_embedding(recycle_latents)
                recycle_condition = recycle_condition.flatten(2).transpose(1, 2)
                recycle_condition = self.sr_proj(recycle_condition)
                if recycle_condition.shape != hidden_states.shape:
                    raise ValueError(
                        "Recycle condition tokens must match Wan latent tokens: "
                        f"recycle={tuple(recycle_condition.shape)}, "
                        f"latent={tuple(hidden_states.shape)}."
                    )
                hidden_states = hidden_states + recycle_condition
            elif recycle_latents is not None:
                raise ValueError(
                    "recycle_latents was provided while recycle_enabled=False."
                )
        block_masks: tuple[BlockMask | None, ...] = (None,) * len(self.blocks)
        if streaming_state is None:
            block_masks = self._get_block_causal_attention_masks(
                device=hidden_states.device,
                num_frames=post_patch_num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
            )

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if streaming_state is None or streaming_state.conditioning_cache is None:
            temb, timestep_proj, encoder_hidden_states = self._embed_conditioning(
                timestep,
                encoder_hidden_states,
            )
        else:
            if timestep.ndim == 2:
                token_count = timestep.shape[1]
            elif timestep.ndim == 1:
                token_count = 1
            else:
                raise ValueError(
                    "Streaming timestep must have shape [batch] or [batch, tokens]."
                )
            temb, timestep_proj, encoder_hidden_states = (
                self._expand_streaming_conditioning(
                    streaming_state=streaming_state,
                    batch_size=batch_size,
                    token_count=token_count,
                )
            )

        # 4. Transformer blocks
        extract_layer_set = set(extract_layers or ())
        last_extract_layer = max(extract_layer_set, default=None)
        extracted_hidden_states: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            self_attn_cache = (
                streaming_state.block_caches[index]
                if streaming_state is not None
                else None
            )
            cross_key_value = None
            if (
                streaming_state is not None
                and streaming_state.conditioning_cache is not None
            ):
                cross_key_value = (
                    streaming_state.conditioning_cache
                    .cross_attention_caches[index].key_value
                )
                if cross_key_value is None:
                    raise RuntimeError("Streaming cross-attention cache was not prepared.")

            # Snapshot immutable history tensors, not the live mutable cache.
            # Cache attributes may advance before backward replays this block.
            block_args = (
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                block_masks[index],
                self_attn_cache.key_value if self_attn_cache is not None else None,
                cross_key_value,
                False,
                post_patch_num_frames if streaming_state is not None else None,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_output = self._gradient_checkpointing_func(block, *block_args)
            else:
                block_output = block(*block_args)
            if self_attn_cache is not None:
                hidden_states, key, value = block_output
                # Commit once per logical forward, outside AC/compile/FSDP block.
                # Detach before retention so cache bookkeeping builds no graph.
                self_attn_cache.update(
                    key.detach(), value.detach(), num_frames=post_patch_num_frames
                )
            else:
                hidden_states = block_output
            if index in extract_layer_set:
                extracted_hidden_states.append(hidden_states)
            if last_extract_layer is not None and index >= last_extract_layer:
                break

        if extract_layers is not None:
            return extracted_hidden_states

        # 5. Output norm, projection & unpatchify
        assert temb.ndim == 3
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (
            self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if streaming_state is not None:
            streaming_state.frame_index += post_patch_num_frames

        return output
