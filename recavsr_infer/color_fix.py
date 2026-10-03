"""Framewise color correction from the StableSR paper (Sec. 3.1, Eqs. 2-5).

Reimplemented for streaming RGB video; no additional models or dependencies.
References:
    https://arxiv.org/abs/1703.06868
    https://arxiv.org/abs/2305.07015
    https://github.com/IceClear/StableSR/blob/main/scripts/wavelet_color_fix.py
    https://github.com/XPixelGroup/DiffBIR/blob/main/diffbir/utils/common.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _adain(content, reference):
    # Spatial statistics only: each frame and RGB channel is independent.
    # Match the reference implementation's sample variance except for 1x1 images.
    correction = int(content.shape[-2] * content.shape[-1] > 1)
    content_var, content_mean = torch.var_mean(
        content, dim=(-2, -1), correction=correction, keepdim=True
    )
    reference_var, reference_mean = torch.var_mean(
        reference, dim=(-2, -1), correction=correction, keepdim=True
    )
    gain = ((reference_var + 1e-5) / (content_var + 1e-5)).sqrt()
    return (content - content_mean) * gain + reference_mean


def _wavelet(content, reference):
    taps = content.new_tensor([1.0, 2.0, 1.0])
    kernel = (torch.outer(taps, taps) / 16).expand(3, 1, 3, 3).contiguous()
    low_content, low_reference = content, reference
    content_high = torch.zeros_like(content)
    # Match the dataset inference protocol: dilations 1, 2, 4, 8, 16.
    for radius in (1, 2, 4, 8, 16):
        padding = (radius,) * 4
        next_content = F.conv2d(
            F.pad(low_content, padding, mode="replicate"),
            kernel,
            dilation=radius,
            groups=3,
        )
        content_high.add_(low_content - next_content)
        low_content = next_content
        low_reference = F.conv2d(
            F.pad(low_reference, padding, mode="replicate"),
            kernel,
            dilation=radius,
            groups=3,
        )
    # Keep the per-level sum: telescoping changes floating-point rounding.
    return content_high + low_reference


def adain_color_fix(decoded: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """AdaIN correction for matching [B,3,T,H,W] RGB tensors in [-1,1].

    Pass cropped, valid frames and the corresponding upsampled LQ reference.
    Computation uses float32 in [0,1]; clipping/quantization happens downstream.
    """
    if decoded.ndim != 5 or decoded.shape[1] != 3 or decoded.shape != reference.shape:
        raise ValueError("Color correction expects matching [B,3,T,H,W] tensors.")
    if decoded.device != reference.device:
        raise ValueError("Decoded frames and color reference must share a device.")
    batch, channels, frames, height, width = decoded.shape
    # Flatten time into batch, never into the spatial statistics or filtering.
    content = decoded.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    source = reference.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    with torch.autocast(device_type=decoded.device.type, enabled=False):
        content = (content.float() + 1) * 0.5
        source = (source.float() + 1) * 0.5
        corrected = _adain(content, source)
        corrected = corrected * 2 - 1
    return corrected.reshape(batch, frames, channels, height, width).permute(
        0, 2, 1, 3, 4
    )


@torch.inference_mode()
def wavelet_color_fix(decoded: torch.Tensor, input_rgb: np.ndarray) -> np.ndarray:
    """Match the dataset wavelet protocol, returning RGB uint8 [T,H,W,3].

    decoded: cropped valid [1,3,T,H,W] frames in [-1,1].
    input_rgb: matching original, unpadded uint8 LQ frames [T,h,w,3].
    The bicubic reference is independent of the model's input preprocessing.
    """
    if decoded.ndim != 5 or decoded.shape[:2] != (1, 3):
        raise ValueError("Wavelet correction expects decoded [1,3,T,H,W] frames.")
    if (
        not isinstance(input_rgb, np.ndarray)
        or input_rgb.dtype != np.uint8
        or input_rgb.ndim != 4
        or input_rgb.shape[0] != decoded.shape[2]
        or input_rgb.shape[-1] != 3
    ):
        raise ValueError(
            "Wavelet reference must be matching uint8 LQ [T,h,w,3] frames."
        )
    with torch.autocast(device_type=decoded.device.type, enabled=False):
        content = decoded[0].permute(1, 0, 2, 3).float().add(1).mul(0.5).clamp_(0, 1)
        reference = (
            torch.from_numpy(np.ascontiguousarray(input_rgb))
            .permute(0, 3, 1, 2)
            .to(device=decoded.device, dtype=torch.float32)
            .div_(255.0)
        )
        reference = F.interpolate(
            reference, size=content.shape[-2:], mode="bicubic", align_corners=False
        ).clamp_(0, 1)
        fixed = _wavelet(content, reference).clamp_(0, 1)
        # Quantize directly in [0,1], without a rounding-sensitive [-1,1] round trip.
        return (
            fixed.mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        )
