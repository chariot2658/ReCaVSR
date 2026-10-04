"""Streaming CPU video I/O with blockwise model input; no cloud/runtime SDKs."""

from __future__ import annotations

import itertools
import math
import re
import subprocess
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


def _image_names(path: Path) -> list[Path]:
    # Numeric frame order works for both 1.png/2.png/10.png and zero-padded names.
    return sorted(
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


def open_video(
    path: str | Path, *, fps: float | None = None, limit: int | None = None
) -> tuple[Iterator[np.ndarray], Fraction]:
    """Return a lazy iterator of RGB uint8 [H,W,3] frames and the playback rate.

    --fps changes playback, not sampling. Frames are decoded on demand, so CPU
    memory does not grow with clip length.
    """
    path = Path(path)
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("FPS must be finite and positive.")
    if limit is not None and limit < 1:
        raise ValueError("Frame limit must be positive.")
    if path.is_dir():
        names = _image_names(path)[:limit]
        rate = Fraction(str(fps or 25))

        def images():
            for name in names:
                bgr = cv2.imread(str(name))
                if bgr is None:
                    raise ValueError(f"Unreadable image: {name}")
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frames = images()
    else:
        with av.open(str(path)) as probe:
            if not probe.streams.video:
                raise ValueError(f"Input contains no video stream: {path}")
            stream = probe.streams.video[0]
            average, width, height = stream.average_rate, stream.width, stream.height
        rate = Fraction(str(fps)) if fps else average
        if rate is None:
            raise ValueError("Input has no average frame rate; provide --fps.")
        frames = ffmpeg_rgb_frames(path, width, height, limit=limit)
    if rate <= 0:
        raise ValueError("Input must have a positive frame rate.")
    return frames, rate


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


def read_video(
    path: str | Path, *, fps: float | None = None, limit: int | None = None
) -> tuple[np.ndarray, Fraction]:
    """Read RGB uint8 [T,H,W,3] frames into memory (small clips and tests)."""
    iterator, rate = open_video(path, fps=fps, limit=limit)
    frames = list(iterator)
    if not frames or len({f.shape for f in frames}) != 1:
        raise ValueError("Input must contain equally-sized RGB frames.")
    return np.stack(frames), rate


def iter_blocks(frames: Iterator[np.ndarray]) -> Iterator[tuple[int, np.ndarray, int]]:
    """Yield (start, valid RGB frames [V,H,W,3], padded block length) lazily."""
    start, shape = 0, None
    while True:
        block = PREFIX_RGB_FRAMES if start == 0 else BODY_RGB_FRAMES
        chunk = list(itertools.islice(frames, block))
        if not chunk:
            if start == 0:
                raise ValueError("A video must contain at least one frame.")
            return
        shape = shape or chunk[0].shape
        if any(f.shape != shape for f in chunk):
            raise ValueError("Input must contain equally-sized RGB frames.")
        yield start, np.stack(chunk), block
        start += len(chunk)
        if len(chunk) < block:
            return


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


def prepare_block(frames, start, valid, block, *, device, scale=4.0, size=None):
    """Upsample one LQ block to the model raster; size=(H, W) overrides scale.

    A non-uniform size (e.g. anamorphic DVD to square pixels) only changes this
    bilinear pre-upsample; the model always works at the output raster.
    """
    # The final chunk repeats its last frame; only valid frames are written out.
    chunk = frames[start : start + valid]
    if start < 0 or len(chunk) != valid or valid < 1 or valid > block:
        raise ValueError("Invalid temporal block.")
    height, width = frames.shape[1:3]
    srh, srw = size or scaled_size(height, width, scale)
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


class VideoWriter:
    """Encode constant-FPS H.264 without audio, one frame batch at a time."""

    def __init__(self, path, rate, height, width, *, crf=18, preset="medium"):
        self.output = av.open(str(path), mode="w")
        self.stream = self.output.add_stream("libx264", rate=rate)
        self.stream.width, self.stream.height = width, height
        # H.264 4:2:0 cannot encode odd dimensions; odd rasters keep all pixels.
        self.stream.pix_fmt = video_pixel_format(height, width)
        self.stream.options = {"crf": str(crf), "preset": preset}
        self.frames = 0

    def write(self, frames: np.ndarray) -> None:
        for array in frames:
            for packet in self.stream.encode(
                av.VideoFrame.from_ndarray(array, format="rgb24")
            ):
                self.output.mux(packet)
        self.frames += len(frames)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()


def write_video(path: str | Path, frames: np.ndarray, rate: Fraction) -> None:
    """Encode constant-FPS H.264 without audio; odd rasters retain all pixels."""
    if not len(frames):
        raise ValueError("Cannot write an empty video.")
    writer = VideoWriter(path, rate, frames.shape[1], frames.shape[2])
    writer.write(frames)
    writer.close()
