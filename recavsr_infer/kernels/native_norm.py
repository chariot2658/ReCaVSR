"""Opaque native reductions for reproducible compiled BF16 inference."""

import torch
from torch.nn import functional as F


@torch.library.custom_op("recavsr_math::layer_norm", mutates_args=())
def native_layer_norm(
    x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float
) -> torch.Tensor:
    return F.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(x.dtype)


@native_layer_norm.register_fake
def _layer_fake(x, weight, bias, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("recavsr_math::rms_norm", mutates_args=())
def native_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


@native_rms_norm.register_fake
def _rms_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
