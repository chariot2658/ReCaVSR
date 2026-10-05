"""Optional lossy FP8 DiT matmuls, retaining BF16 attention and cache tensors.

FP8 weights are stored as opaque BF16 parameter views so BlockStreamer can copy
them with its existing single-dtype arena. These views are never used for BF16
arithmetic. The original checkpoint is unchanged; small FP32 scales stay resident.
"""

from __future__ import annotations

import torch
from torch import nn

FP8_MAX = 448.0


class FP8Linear(nn.Module):
    """Tensor-scaled E4M3 weights and dynamically scaled E4M3 activations.

    This is inference-only. Convert after loading and moving the original model;
    casting the packed parameter to another dtype would corrupt its stored bytes.
    """

    @torch.inference_mode()
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.weight.device.type != "cuda" or linear.weight.dtype != torch.bfloat16:
            raise ValueError("FP8 conversion requires loaded CUDA BF16 linears.")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 linear dimensions must be multiples of 16.")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight = linear.weight.detach().float()
        scale = weight.abs().amax().clamp_min(1e-12) / FP8_MAX
        quantized = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        self.weight_packed = nn.Parameter(quantized.view(torch.bfloat16), requires_grad=False)
        self.bias = linear.bias
        self.register_buffer("weight_scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = x.reshape(-1, self.in_features)
        scale = matrix.float().abs().amax().clamp_min(1e-12) / FP8_MAX
        quantized = (matrix.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        weight = self.weight_packed.view(torch.float8_e4m3fn).t()
        out = torch._scaled_mm(
            quantized, weight, scale_a=scale, scale_b=self.weight_scale,
            bias=self.bias, out_dtype=x.dtype, use_fast_accum=False,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


def quantize_block_linears(block: nn.Module) -> None:
    """Convert forward linears after cross-attention K/V have been consumed."""
    for module in list(block.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, FP8Linear(child))
