"""Native RMSNorm/GEMM boundaries preserve source autocast and rounding."""

import torch
from torch.nn import functional as F

from ..models.flashdecoder.flashdecoder_wan22 import FlashDecoder


@torch.library.custom_op("recavsr_flash_math::rms_norm", mutates_args=())
def flash_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # CUDA autocast promotes source RMSNorm inputs to FP32. Preserve that boundary.
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps)


@flash_rms_norm.register_fake
def _fake(x, weight, eps):
    return torch.empty_like(
        x,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )


class NativeFlashRMSNorm(torch.nn.RMSNorm):
    def forward(self, x):
        if torch.compiler.is_compiling():
            return flash_rms_norm(x, self.weight, self.eps)
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


@torch.library.custom_op("recavsr_flash_math::linear", mutates_args=())
def flash_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.linear(x, weight, bias)


@flash_linear.register_fake
def _linear_fake(x, weight, bias):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class NativeFlashLinear(torch.nn.Linear):
    # Keep GEMM output materialization: unrestricted fusion changed BF16 rounding.
    # Casts stay outside the custom op so inference freezing can fold static weights.
    def forward(self, x):
        if torch.compiler.is_compiling():
            return flash_linear(
                x.bfloat16(),
                self.weight.bfloat16(),
                None if self.bias is None else self.bias.bfloat16(),
            )
        return F.linear(x, self.weight, self.bias)


@torch.library.custom_op("recavsr_flash_math::encode_lr", mutates_args=())
def flash_encode_lr(
    lr: torch.Tensor,
    stem_weight: torch.Tensor,
    stem_bias: torch.Tensor,
    early_weight: torch.Tensor,
    late_weight: torch.Tensor,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve the source pixel-unshuffle layout and its Linear dispatch together.
    # Layout rewriting can change whether the BF16 bias is fused into GEMM.
    with torch.autocast(device_type=lr.device.type, enabled=False):
        batch, _, frames, height, width = lr.shape
        pixels = lr.transpose(1, 2).reshape(batch * frames, 3, height, width)
        packed = F.pixel_unshuffle(pixels, 16).permute(0, 2, 3, 1)
        features = F.linear(packed.bfloat16(), stem_weight, stem_bias)
        height, width = height // 16, width // 16
        features = features.reshape(batch, frames, height, width, 48)
        if first:
            features = torch.cat(
                (features[:, :1].expand(-1, 3, -1, -1, -1), features), dim=1
            )
        groups = features.shape[1] // 4
        grouped = (
            features.reshape(batch, groups, 4, height, width, 48)
            .permute(0, 1, 3, 4, 2, 5)
            .reshape(batch, groups * height * width, 192)
        )
        return (
            F.linear(grouped, early_weight),
            F.linear(features.reshape(batch, -1, 48), late_weight),
        )


@flash_encode_lr.register_fake
def _lr_fake(lr, stem_weight, stem_bias, early_weight, late_weight, first):
    batch, _, frames, height, width = lr.shape
    tokens = (height // 16) * (width // 16)
    effective_frames = frames + (3 if first else 0)
    return (
        stem_weight.new_empty(
            (batch, effective_frames // 4 * tokens, early_weight.shape[0])
        ),
        stem_weight.new_empty((batch, effective_frames * tokens, late_weight.shape[0])),
    )


class NativeFlashLRDecoder(FlashDecoder):
    def _encode_lr(self, latents, lr_up, *, first_chunk):
        if torch.compiler.is_compiling():
            return flash_encode_lr(
                lr_up,
                self.lr_stem.weight.bfloat16(),
                self.lr_stem.bias.bfloat16(),
                self.lr_early_projection.weight.bfloat16(),
                self.lr_late_projection.weight.bfloat16(),
                first_chunk,
            )
        return super()._encode_lr(latents, lr_up, first_chunk=first_chunk)


def preserve_native_norms(model):
    """Install inference math boundaries without replacing any master parameter."""
    if getattr(model, "_flash_native_norms", False):
        return
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.RMSNorm:
                with torch.device("meta"):
                    replacement = NativeFlashRMSNorm(
                        child.normalized_shape, eps=child.eps
                    )
                replacement.weight = child.weight
                replacement.train(child.training)
                setattr(parent, name, replacement)
            elif type(child) is torch.nn.Linear:
                with torch.device("meta"):
                    replacement = NativeFlashLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                    )
                replacement.weight = child.weight
                replacement.bias = child.bias
                replacement.train(child.training)
                setattr(parent, name, replacement)
    if type(model) is FlashDecoder:
        model.__class__ = NativeFlashLRDecoder
    model._flash_native_norms = True
