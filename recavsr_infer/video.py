"""Whole-video CPU I/O with blockwise model input; no cloud/runtime SDKs."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .runtime.constants import BODY_RGB_FRAMES, PREFIX_RGB_FRAMES, SPATIAL_TOKEN_STRIDE


def validate_scale(scale):
    """Return a finite positive spatial scale (time is never resampled)."""
    if isinstance(scale, bool):
        raise ValueError("Scale must be a finite positive number.")
    try:
        scale = float(scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Scale must be a finite positive number.") from error
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be a finite positive number.")
    return scale


def scaled_size(height, width, scale=4.0):
    """Round each scaled dimension half-up, shared by preparation and cropping."""
    scale = validate_scale(scale)
    if height < 1 or width < 1:
        raise ValueError("Input dimensions must be positive.")
    dimensions = (height * scale, width * scale)
    if any(not math.isfinite(x) or x > 2**31 - 1 for x in dimensions):
        raise ValueError("Scaled dimensions exceed the supported integer range.")
    result = tuple(math.floor(x + 0.5) for x in dimensions)
    if min(result) < 1:
        raise ValueError("Scale produces an empty output dimension.")
    return result


def video_pixel_format(height, width):
    # H.264 4:2:0 cannot encode odd dimensions. Preserve the exact raster using 4:4:4.
    return "yuv420p" if height % 2 == 0 and width % 2 == 0 else "yuv444p"


def read_video(
    path: str | Path, *, fps: float | None = None, limit: int | None = None
) -> tuple[np.ndarray, Fraction]:
    """Read RGB uint8 [T,H,W,3] frames; --fps changes playback, not sampling."""
    path = Path(path)
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("FPS must be finite and positive.")
    if limit is not None and limit < 1:
        raise ValueError("Frame limit must be positive.")
    if path.is_dir():
        # Numeric frame order works for both 1.png/2.png/10.png and zero-padded names.
        names = sorted(
            (
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            ),
            key=lambda p: (
                tuple(
                    int(part) if part.isdigit() else part.lower()
                    for part in re.split(r"(\d+)", p.name)
                ),
                p.name,
            ),
        )
        if limit is not None:
            names = names[:limit]
        frames = []
        for name in names:
            bgr = cv2.imread(str(name))
            if bgr is None:
                raise ValueError(f"Unreadable image: {name}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rate = Fraction(str(fps or 25))
    else:
        frames = []
        with av.open(str(path)) as video:
            if not video.streams.video:
                raise ValueError(f"Input contains no video stream: {path}")
            stream = video.streams.video[0]
            rate = Fraction(str(fps)) if fps else stream.average_rate
            if rate is None:
                raise ValueError("Input has no average frame rate; provide --fps.")
            for frame in video.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                if limit is not None and len(frames) >= limit:
                    break
    if not frames or len({f.shape for f in frames}) != 1 or rate <= 0:
        raise ValueError(
            "Input must contain equally-sized RGB frames and a positive frame rate."
        )
    return np.stack(frames), rate


def block_schedule(total: int) -> Iterator[tuple[int, int, int]]:
    """Yield (start, valid RGB frames, padded RGB frames) for one continuous session."""
    if total < 1:
        raise ValueError("A video must contain at least one frame.")
    start = 0
    while start < total:
        block = PREFIX_RGB_FRAMES if start == 0 else BODY_RGB_FRAMES
        valid = min(block, total - start)
        yield start, valid, block
        start += valid


def prepare_block(frames, start, valid, block, *, device, scale=4.0):
    # The final chunk repeats its last frame; only valid frames are written out.
    chunk = frames[start : start + valid]
    if start < 0 or len(chunk) != valid or valid < 1 or valid > block:
        raise ValueError("Invalid temporal block.")
    height, width = frames.shape[1:3]
    srh, srw = scaled_size(height, width, scale)
    enlarged = np.stack(
        [cv2.resize(f, (srw, srh), interpolation=cv2.INTER_LINEAR) for f in chunk]
    )
    x = (
        torch.from_numpy(enlarged)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    x = x / 127.5 - 1.0
    if valid < block:
        x = torch.cat((x, x[:, :, -1:].expand(-1, -1, block - valid, -1, -1)), 2)
    ph, pw = (-srh) % SPATIAL_TOKEN_STRIDE, (-srw) % SPATIAL_TOKEN_STRIDE
    if pw:
        x = F.pad(x, (0, pw, 0, 0, 0, 0), mode="reflect" if pw < srw else "replicate")
    if ph:
        x = F.pad(x, (0, 0, 0, ph, 0, 0), mode="reflect" if ph < srh else "replicate")
    return x


def rgb_uint8(decoded: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Crop decoded [1,3,T,H,W] values in [-1,1] into RGB uint8 [T,H,W,3]."""
    return (
        ((decoded[0, :, :, :height, :width].float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def write_video(path: str | Path, frames: np.ndarray, rate: Fraction) -> None:
    """Encode constant-FPS H.264 without audio; odd rasters retain all pixels."""
    if not len(frames):
        raise ValueError("Cannot write an empty video.")
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = video_pixel_format(stream.height, stream.width)
        stream.options = {"crf": "18", "preset": "medium"}
        for array in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(array, format="rgb24")
            ):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
