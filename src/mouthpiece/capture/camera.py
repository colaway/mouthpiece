"""Webcam capture at a fixed target frame rate."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType

import cv2
import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[np.uint8]  # H x W x 3, BGR


@dataclass(frozen=True)
class TimedFrame:
    index: int          # index in the resampled (target-fps) stream
    timestamp: float    # seconds, monotonic clock
    image: Frame


class FrameRateLimiter:
    """Drop frames so a stream arriving at >= target fps is emitted at target fps.

    Webcams on macOS usually deliver 30 fps even when 25 is requested. The model was
    trained on 25 fps video, so we keep a frame whenever its timestamp reaches the next
    25 fps slot. Slots that are missed (camera slower than target) are skipped, not
    duplicated, and the schedule re-anchors after a long gap.
    """

    def __init__(self, target_fps: float, max_gap_s: float = 0.5) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.period = 1.0 / target_fps
        self.max_gap_s = max_gap_s
        self._next: float | None = None

    def reset(self) -> None:
        self._next = None

    def accept(self, t: float) -> bool:
        if self._next is None or t - self._next > self.max_gap_s:
            self._next = t + self.period
            return True
        # Small tolerance so jitter around the slot boundary doesn't drop frames.
        if t + 0.25 * self.period >= self._next:
            self._next += self.period
            if self._next < t:  # fell behind: realign to now
                self._next = t + self.period
            return True
        return False


def resample_indices(timestamps: list[float], target_fps: float) -> list[int]:
    """Indices of frames to keep so that `timestamps` are resampled to `target_fps`.

    For each target slot k/target_fps (k = 0, 1, …) pick the nearest source frame.
    Used for video files whose native rate is not 25 fps.
    """
    if not timestamps:
        return []
    t0, t_end = timestamps[0], timestamps[-1]
    n = int(np.floor((t_end - t0) * target_fps + 1e-6)) + 1
    ts = np.asarray(timestamps)
    slots = t0 + np.arange(n) / target_fps
    return [int(np.argmin(np.abs(ts - s))) for s in slots]


class Camera:
    """OpenCV/AVFoundation camera yielding frames at the target fps."""

    def __init__(self, index: int = 0, fps: int = 25, width: int = 640, height: int = 480) -> None:
        self.index, self.fps, self.width, self.height = index, fps, width, height
        self._cap: cv2.VideoCapture | None = None
        self._limiter = FrameRateLimiter(fps)

    def open(self) -> Camera:
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.index}. "
                "Run `uv run mouthpiece doctor --probe-camera`."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._limiter.reset()
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def native_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    def frames(self) -> Iterator[TimedFrame]:
        if self._cap is None:
            raise RuntimeError("Camera not opened")
        idx = 0
        failures = 0
        while True:
            ok, img = self._cap.read()
            t = time.monotonic()
            if not ok or img is None:
                failures += 1
                if failures > 50:
                    raise RuntimeError("Camera stopped delivering frames")
                continue
            failures = 0
            if self._limiter.accept(t):
                yield TimedFrame(idx, t, np.asarray(img, dtype=np.uint8))
                idx += 1


def read_video(path: str, target_fps: float | None = 25.0) -> tuple[list[Frame], float]:
    """Read a video file (BGR frames), resampled to `target_fps` if given.

    Returns (frames, fps_of_returned_frames).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[Frame] = []
    stamps: list[float] = []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            stamps.append(ms / 1000.0 if ms > 0 or not stamps else len(stamps) / src_fps)
            frames.append(np.asarray(img, dtype=np.uint8))
    finally:
        cap.release()
    if target_fps is None or abs(src_fps - target_fps) < 0.01:
        return frames, float(src_fps)
    keep = resample_indices(stamps, target_fps)
    return [frames[i] for i in keep], float(target_fps)
