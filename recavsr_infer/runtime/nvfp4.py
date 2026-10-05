"""Optional lossy NVFP4 DiT linears for Blackwell GPUs.

Weights and activations use E2M1 with an E4M3 scale per 16 values. Attention,
caches and conditioning retain their original precision. Weight and scale bytes
are opaque BF16 parameters for BlockStreamer; never dtype-cast those views.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def validate_support(device: torch.device) -> None:
    """Fail before loading the model when the required API/hardware is absent."""
    if not hasattr(torch, "_scaled_mm_v2") or not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("NVFP4 requires PyTorch with _scaled_mm_v2 and float4_e2m1fn_x2 support.")
    if torch.cuda.get_device_capability(device) < (10, 0):
        raise ValueError("NVFP4 DiT requires an NVIDIA Blackwell GPU or later.")


@torch.library.custom_op("recavsr_nvfp4::linear", mutates_args=())
def nvfp4_mm(
    values: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Inductor's v2 lowering does not support swizzled scales. Keep this native
    # GEMM opaque while compiling the activation quantization around it.
    # Recipe 2 = BlockWise1x16; swizzle 1 = SWIZZLE_32_4_4.
    return torch._scaled_mm_v2(
        values.view(torch.float4_e2m1fn_x2),
        weights.view(torch.float4_e2m1fn_x2).t(),
        [scales.view(torch.float8_e4m3fn)], [2], [1],
        [weight_scales.view(torch.float8_e4m3fn)], [2], [1],
        bias, torch.bfloat16,
    )


@nvfp4_mm.register_fake
def _nvfp4_mm_fake(values, weights, scales, weight_scales, bias):
    return torch.empty(
        (values.shape[0], weights.shape[0]), device=values.device, dtype=torch.bfloat16
    )


def quantize(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack E2M1 pairs and swizzle padded block scales for native NVFP4 GEMM."""
    m, k = matrix.shape
    blocks = matrix.float().reshape(m, k // 16, 16)
    scales = (blocks.abs().amax(-1) / 6).clamp_min(2**-9).to(torch.float8_e4m3fn)
    z = blocks / scales.float().unsqueeze(-1)
    a = z.abs()
    # E2M1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6; ties round to even.
    code = (a > .25).to(torch.uint8)
    for threshold, inclusive in ((.75, True), (1.25, False), (1.75, True),
                                 (2.5, False), (3.5, True), (5., False)):
        code = code + (a >= threshold if inclusive else a > threshold).to(torch.uint8)
    code = code | ((z < 0).to(torch.uint8) * 8)
    code = code.reshape(m, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    pm, pk = (m + 127) // 128 * 128, ((k // 16 + 3) // 4) * 4
    padded = F.pad(scales.view(torch.uint8), (0, pk - k // 16, 0, pm - m))
    blocked = padded.reshape(pm // 128, 4, 32, pk // 4, 4)
    blocked = blocked.permute(0, 3, 2, 1, 4).contiguous().flatten()
    return packed, blocked


class NVFP4Linear(nn.Module):
    """Inference-only block-scaled FP4 with dynamically quantized activations."""

    @torch.inference_mode()
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.weight.device.type != "cuda" or linear.weight.dtype != torch.bfloat16:
            raise ValueError("NVFP4 conversion requires loaded CUDA BF16 linears.")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("NVFP4 linear dimensions must be multiples of 16.")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight, scales = quantize(linear.weight.detach())
        self.weight_packed = nn.Parameter(weight.view(torch.bfloat16), requires_grad=False)
        self.scales_packed = nn.Parameter(scales.view(torch.bfloat16), requires_grad=False)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError("NVFP4 linears require BF16 inputs and produce BF16 outputs.")
        matrix = x.reshape(-1, self.in_features)
        values, scales = quantize(matrix)
        out = nvfp4_mm(values, self.weight_packed, scales, self.scales_packed, self.bias)
        return out.reshape(*x.shape[:-1], self.out_features)


def quantize_block_linears(block: nn.Module) -> None:
    """Convert forward linears after cross-attention K/V have been consumed."""
    for module in list(block.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, NVFP4Linear(child))
