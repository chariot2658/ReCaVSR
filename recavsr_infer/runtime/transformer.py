"""Pure-tensor transformer blocks with per-video rolling state outside compile."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..kernels.local_attention import paired_attention, sdpa_attention
from ..kernels.native_norm import native_layer_norm, native_rms_norm
from .cache import FixedKVCache
from .constants import (
    BODY_LATENT_FRAMES,
    BODY_RGB_FRAMES,
    DEFAULT_SPATIAL_WINDOW,
    INFERENCE_TIMESTEP,
    PREFIX_LATENT_FRAMES,
    PREFIX_RGB_FRAMES,
    SPATIAL_TOKEN_STRIDE,
    VAE_SPATIAL_STRIDE,
)
from .rope import RollingRoPE, apply_rotary


class PreparedBlock(nn.Module):
    """Compile-friendly block: fixed conditioning inside, mutable KV state outside."""

    def __init__(
        self,
        block,
        conditioning,
        rope,
        *,
        window=DEFAULT_SPATIAL_WINDOW,
        native_norms=False,
    ):
        super().__init__()
        self.block = block
        self.height, self.width = rope.height, rope.width
        self.window = tuple(window)
        self.native_norms = native_norms
        # Window and grid dimensions are in tokens. A covering window uses SDPA.
        self.local = self.height > window[0] or self.width > window[1]
        # The released model uses one fixed timestep and prompt for the whole video.
        modulation = (
            block.scale_shift_table.unsqueeze(0) + conditioning.timestep_proj.float()
        )
        self.register_buffer("modulation", modulation)
        cross_k, cross_v = block.attn2.precompute_cross_attention_kv(
            conditioning.encoder_hidden_states
        )
        self.register_buffer("cross_k", cross_k)
        self.register_buffer("cross_v", cross_v)
        self.register_buffer("spatial_cos", rope.spatial_cos)
        self.register_buffer("spatial_sin", rope.spatial_sin)
        self.register_buffer("temporal_cos", rope.temporal_cos)
        self.register_buffer("temporal_sin", rope.temporal_sin)

    def layer_norm(self, module, x):
        if isinstance(module, nn.Identity):
            return x
        if self.native_norms:
            return native_layer_norm(x, module.weight, module.bias, module.eps)
        return module(x)

    def rms_norm(self, module, x):
        return (
            native_rms_norm(x, module.weight, module.eps)
            if self.native_norms
            else module(x)
        )

    def spatial(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1, self.height * self.width, shape[2], shape[3])
        return apply_rotary(x, self.spatial_cos, self.spatial_sin).reshape(shape)

    def forward(
        self, hidden, past_key, past_value, slots, past_positions, query_positions
    ):
        """Return hidden states and spatially rotated KV for future chunks."""
        # Temporal RoPE is applied inside attention, relative to the rolling origin;
        # it must not be baked into the persistent KV cache.
        b = self.block
        shift, scale, gate, cshift, cscale, cgate = [
            part.squeeze(2) for part in self.modulation.chunk(6, dim=2)
        ]
        norm = (self.layer_norm(b.norm1, hidden.float()) * (1 + scale) + shift).to(
            hidden.dtype
        )
        a = b.attn1
        if a.fused_projections:
            q, k, v = a.to_qkv(norm).chunk(3, -1)
        else:
            q, k, v = a.to_q(norm), a.to_k(norm), a.to_v(norm)
        q = self.spatial(self.rms_norm(a.norm_q, q).unflatten(2, (a.heads, -1)))
        k = self.spatial(self.rms_norm(a.norm_k, k).unflatten(2, (a.heads, -1)))
        v = v.unflatten(2, (a.heads, -1))
        attention = paired_attention if self.local else sdpa_attention
        out = attention(
            q,
            k,
            v,
            past_key,
            past_value,
            slots,
            past_positions,
            query_positions,
            self.temporal_cos,
            self.temporal_sin,
            self.height,
            self.width,
            *self.window,
        )
        out = a.to_out[1](a.to_out[0](out.flatten(2).to(q.dtype)))
        hidden = (hidden.float() + out * gate).to(hidden.dtype)
        a = b.attn2
        norm = self.layer_norm(b.norm2, hidden.float()).to(hidden.dtype)
        q = self.rms_norm(a.norm_q, a.to_q(norm)).unflatten(2, (a.heads, -1))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            self.cross_k.transpose(1, 2),
            self.cross_v.transpose(1, 2),
            dropout_p=0.0,
        ).transpose(1, 2)
        hidden = hidden + a.to_out[1](a.to_out[0](out.flatten(2).to(q.dtype)))
        norm = (self.layer_norm(b.norm3, hidden.float()) * (1 + cscale) + cshift).to(
            hidden.dtype
        )
        hidden = (hidden.float() + b.ffn(norm).float() * cgate).to(hidden.dtype)
        return hidden, k, v


class TransformerSession:
    """State for one video: LQ convolution history, routed KV, and recycled latents.

    Call step consecutively on the prefix and body chunks. Construct a new session
    for a new video or geometry; attention chunks are never independent clips.
    """

    @torch.inference_mode()
    def __init__(
        self,
        model,
        prompt,
        height,
        width,
        *,
        seed=42,
        window=DEFAULT_SPATIAL_WINDOW,
        compile_blocks=False,
        compile_mode="default",
        native_norms=False,
    ):
        if (
            height % SPATIAL_TOKEN_STRIDE
            or width % SPATIAL_TOKEN_STRIDE
            or min(height, width) < SPATIAL_TOKEN_STRIDE
        ):
            raise ValueError("Transformer geometry must be a positive multiple of 32.")
        if len(window) != 2 or min(window) < 1:
            raise ValueError("Spatial windows must be positive.")
        router = model._static_kv_router
        if router is None or len(router.layer_actions) != len(model.blocks):
            raise ValueError(
                "A static KV router action is required for every DiT block."
            )
        self.model, self.height, self.width = model, height, width
        self.use_cudagraphs = compile_blocks and compile_mode in (
            "reduce-overhead",
            "max-autotune",
        )
        self.device = model.patch_embedding.weight.device
        self.dtype = model.patch_embedding.weight.dtype
        self.rope = RollingRoPE(
            model.config.attention_head_dim,
            height // SPATIAL_TOKEN_STRIDE,
            width // SPATIAL_TOKEN_STRIDE,
            device=self.device,
        )
        self.conditioning = model._cache_streaming_conditioning(
            torch.tensor([INFERENCE_TIMESTEP], device=self.device),
            prompt.to(device=self.device, dtype=self.dtype),
        )
        self.cores = [
            PreparedBlock(
                b,
                self.conditioning,
                self.rope,
                window=window,
                native_norms=native_norms,
            ).eval()
            for b in model.blocks
        ]
        if compile_blocks:
            # Finite signatures: prefix, capacities 0/1/2/4/6 and distinct fill states.
            torch._dynamo.config.recompile_limit = 64
            self.cores = [
                torch.compile(
                    b,
                    fullgraph=True,
                    dynamic=False,
                    options={
                        **torch._inductor.list_mode_options(compile_mode),
                        "emulate_precision_casts": True,
                        "force_same_precision": True,
                    },
                )
                for b in self.cores
            ]
        self.caches = [
            FixedKVCache(
                a,
                initial=router.anchor_initial_index,
                stride=router.anchor_stride_latents,
                anchors=router.anchor_capacity_latents,
            )
            for a in router.layer_actions
        ]
        self.lq_state = model.lq_proj.create_streaming_state()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.latent_offset, self.previous_latents = 0, None

    @torch.inference_mode()
    def step(
        self, lq: torch.Tensor, *, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Consume [1,3,T,H,W] normalized RGB and return the next latent block."""
        if self.use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        frames = PREFIX_LATENT_FRAMES if self.latent_offset == 0 else BODY_LATENT_FRAMES
        expected_rgb = PREFIX_RGB_FRAMES if self.latent_offset == 0 else BODY_RGB_FRAMES
        if lq.shape != (1, 3, expected_rgb, self.height, self.width):
            raise ValueError(
                f"Expected LQ block [1,3,{expected_rgb},{self.height},{self.width}]."
            )
        shape = (
            1,
            self.model.config.in_channels,
            frames,
            self.height // VAE_SPATIAL_STRIDE,
            self.width // VAE_SPATIAL_STRIDE,
        )
        if noise is None:
            noise = torch.randn(
                shape, generator=self.generator, device=self.device, dtype=torch.float32
            )
        if (
            noise.shape != shape
            or noise.dtype != torch.float32
            or noise.device != self.device
        ):
            raise ValueError(
                "Explicit noise must match the block in FP32 on the model device."
            )
        m = self.model
        recycle = (
            torch.zeros_like(noise, dtype=self.dtype)
            if self.previous_latents is None
            else self.previous_latents
        )
        hidden = m.patch_embedding(noise.to(self.dtype)).flatten(2).transpose(1, 2)
        hidden = hidden + m.lq_proj.forward_streaming(
            lq.to(self.device, self.dtype), self.lq_state
        )
        hidden = hidden + m.sr_proj(
            m.patch_embedding(recycle).flatten(2).transpose(1, 2)
        )
        origin, positions = self.rope.positions(self.latent_offset, frames)
        dummy = hidden.unflatten(2, (m.config.num_attention_heads, -1))
        for core, cache in zip(self.cores, self.caches):
            history = cache.history(dummy, origin=origin)
            hidden, k, v = core(hidden, *history, positions)
            cache.commit(k, v, frames=frames, start=self.latent_offset)
        shift, scale = (
            m.scale_shift_table.unsqueeze(0) + self.conditioning.temb.unsqueeze(2)
        ).chunk(2, 2)
        hidden = (
            m.norm_out(hidden.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        ).to(self.dtype)
        hidden = m.proj_out(hidden)
        pt, ph, pw = m.config.patch_size
        prediction = hidden.reshape(
            1,
            frames,
            self.height // SPATIAL_TOKEN_STRIDE,
            self.width // SPATIAL_TOKEN_STRIDE,
            pt,
            ph,
            pw,
            -1,
        )
        prediction = (
            prediction.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        result = noise - prediction.float()
        # Own the recycle buffer across steps, including CUDA-graph output reuse.
        self.previous_latents = (
            result[:, :, -BODY_LATENT_FRAMES:].to(self.dtype).clone()
        )
        self.latent_offset += frames
        return result
