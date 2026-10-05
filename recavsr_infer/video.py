"""Streaming CPU video I/O with blockwise model input; no cloud/runtime SDKs."""

from __future__ import annotations

import itertools
import math
import queue
import re
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .frames import (  # noqa: F401
    SWS_ACCURATE,
    ffmpeg_rgb_frames,
    scaled_size,
    validate_scale,
)
from .runtime.constants import BODY_RGB_FRAMES, PREFIX_RGB_FRAMES, SPATIAL_TOKEN_STRIDE


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
    size = size or scaled_size(height, width, scale)
    return block_tensor(enlarge_block(chunk, size), block, device=device)


def enlarge_block(chunk: np.ndarray, size, *, pin: bool = False) -> torch.Tensor:
    """CPU half of prepare_block: bilinear-upsample uint8 frames to size=(H, W).

    Returns contiguous uint8 [1,3,T,H,W], page-locked if pin (for an async upload).
    """
    srh, srw = size
    enlarged = np.stack(
        [cv2.resize(f, (srw, srh), interpolation=cv2.INTER_LINEAR) for f in chunk]
    )
    x = torch.from_numpy(enlarged).permute(3, 0, 1, 2).unsqueeze(0)
    return x.pin_memory() if pin else x.contiguous()


def block_tensor(enlarged: torch.Tensor, block: int, *, device) -> torch.Tensor:
    """Device half of prepare_block: normalize and pad to the block and token stride."""
    valid, srh, srw = enlarged.shape[2:]
    # uint8 -> float32 is exact, so converting after the upload changes nothing.
    x = enlarged.to(device=device, non_blocking=True).to(torch.float32)
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


def prefetch(items: Iterator, depth: int = 2) -> Iterator:
    """Produce items on a background thread, up to depth ahead of the consumer.

    Keeps CPU decoding and resizing off the GPU loop's critical path. Producer
    errors are re-raised in the consumer; closing the consumer stops the producer.
    """
    out: queue.Queue = queue.Queue(depth)
    stop = threading.Event()
    end = object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            for item in items:
                if not put((item, None)):
                    break
            else:
                put((end, None))
        except BaseException as error:
            put((end, error))
        finally:
            if hasattr(items, "close"):
                items.close()  # e.g. kills a decoder subprocess when stopped early

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item, error = out.get()
            if error is not None:
                raise error
            if item is end:
                return
            yield item
    finally:
        stop.set()
        thread.join()


class BackgroundWriter:
    """A VideoWriter that encodes on its own thread, overlapping GPU work."""

    def __init__(self, writer: VideoWriter, depth: int = 4):
        self.writer, self.frames = writer, 0
        self.queue: queue.Queue = queue.Queue(depth)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while (frames := self.queue.get()) is not None:
            if self.error is None:
                try:
                    self.writer.write(frames)
                except BaseException as error:
                    self.error = error  # keep draining so write() never blocks

    def write(self, frames: np.ndarray) -> None:
        if self.error is not None:
            raise self.error
        self.queue.put(frames)
        self.frames += len(frames)

    def _finish(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def close(self) -> None:
        self._finish()
        if self.error is not None:
            raise self.error
        self.writer.close()

    def abort(self) -> None:
        """Stop without flushing; the caller deletes the partial file."""
        self._finish()
        self.writer.output.close()


def write_video(path: str | Path, frames: np.ndarray, rate: Fraction) -> None:
    """Encode constant-FPS H.264 without audio; odd rasters retain all pixels."""
    if not len(frames):
        raise ValueError("Cannot write an empty video.")
    writer = VideoWriter(path, rate, frames.shape[1], frames.shape[2])
    writer.write(frames)
    writer.close()
