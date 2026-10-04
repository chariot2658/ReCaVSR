"""Torch-free frame helpers: output geometry and accurate ffmpeg RGB decoding.

Kept apart from video.py so CPU-only tools (the strip blender) don't import torch,
which costs ~0.9 GiB of commit on Windows.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterator

import numpy as np


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


# swscale's default YUV->RGB path truncates, darkening every decode by ~1 level
# (PyAV's to_ndarray included, and it cannot request other flags). Encoding is
# unbiased, so only decoding goes through the ffmpeg CLI with accurate rounding.
SWS_ACCURATE = "accurate_rnd+full_chroma_int+bitexact"


def ffmpeg_rgb_frames(path, width, height, *, limit=None) -> Iterator[np.ndarray]:
    """Decode a video's frames to RGB uint8 [H,W,3] with accurate rounding."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-sws_flags",
        SWS_ACCURATE,
    ]
    if limit is not None:
        command += ["-frames:v", str(limit)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=size * 4)
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise RuntimeError(f"Truncated frame while decoding {path}.")
            yield np.frombuffer(data, np.uint8).reshape(height, width, 3)
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to decode {path}.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
