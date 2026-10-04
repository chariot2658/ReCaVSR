"""Vertical strips: run frames wider than VRAM allows as overlapping columns.

Each strip is an independent full-length video session, so temporal state and
precision are unchanged; only the image edge inside each overlap is new, and
linear feathering across the overlap hides it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import av
import cv2
import numpy as np

from .runtime.constants import SPATIAL_TOKEN_STRIDE
from .video import ffmpeg_rgb_frames


def strip_columns(
    width: int, strips: int, overlap: int, scale: float
) -> list[tuple[int, int]]:
    """Return equal-width LQ column ranges [x0, x1) covering 0..width.

    The strip width is rounded up so its scaled width is a multiple of the
    token stride (no model padding), and every strip has the same geometry so
    one prepared/compiled session serves all of them.
    """
    if strips < 2:
        return [(0, width)]
    step = 1
    if float(scale).is_integer():
        step = SPATIAL_TOKEN_STRIDE // math.gcd(SPATIAL_TOKEN_STRIDE, int(scale))
    base = math.ceil((width + (strips - 1) * overlap) / strips)
    strip = min(width, -(-base // step) * step)
    starts = [round(i * (width - strip) / (strips - 1)) for i in range(strips)]
    columns = [(x, x + strip) for x in starts]
    for (_, a1), (b0, _) in zip(columns, columns[1:]):
        if a1 - b0 < 1:
            raise ValueError("Strips do not overlap; increase --strip-overlap.")
    return columns


def column_weights(
    columns, scale: float, out_width: int
) -> list[tuple[int, np.ndarray]]:
    """Per strip: (output x offset, 1-D feather weights over its output width)."""
    placed = [(round(x0 * scale), round(x1 * scale)) for x0, x1 in columns]
    weights = []
    for i, (o0, o1) in enumerate(placed):
        w = np.ones(o1 - o0, dtype=np.float32)
        if i > 0:
            ramp = placed[i - 1][1] - o0
            w[:ramp] = np.minimum(w[:ramp], (np.arange(ramp) + 0.5) / ramp)
        if i < len(placed) - 1:
            ramp = o1 - placed[i + 1][0]
            w[-ramp:] = np.minimum(w[-ramp:], (np.arange(ramp)[::-1] + 0.5) / ramp)
        weights.append((o0, w))
    if placed[-1][1] != out_width:
        raise ValueError("Strips must cover the scaled frame exactly.")
    return weights


def _decode(path) -> Iterator[np.ndarray]:
    with av.open(str(path)) as probe:
        stream = probe.streams.video[0]
        width, height = stream.width, stream.height
    return ffmpeg_rgb_frames(path, width, height)


def blend_strips(
    paths, columns, scale, height, width, resize=None
) -> Iterator[np.ndarray]:
    """Yield blended RGB uint8 frames, optionally resized (area) to (W, H)."""
    weights = column_weights(columns, scale, width)
    total = np.zeros(width, dtype=np.float32)
    for o0, w in weights:
        total[o0 : o0 + len(w)] += w
    streams = [_decode(p) for p in paths]
    while True:
        frames = [next(s, None) for s in streams]
        if all(f is None for f in frames):
            return
        if any(f is None for f in frames):
            raise RuntimeError("Strip videos have different frame counts.")
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        for f, (o0, w) in zip(frames, weights):
            if f.shape[:2] != (height, len(w)):
                raise RuntimeError(f"Unexpected strip frame size {f.shape[:2]}.")
            canvas[:, o0 : o0 + len(w)] += f * w[None, :, None]
        canvas /= total[None, :, None]
        out = np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
        if resize is not None and (resize[0], resize[1]) != (width, height):
            out = cv2.resize(out, tuple(resize), interpolation=cv2.INTER_AREA)
        yield out
