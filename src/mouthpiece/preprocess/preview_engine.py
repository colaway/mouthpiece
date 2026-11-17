"""UI-independent preview logic: landmarks -> smoothing -> crop -> stats -> clip recording.

The Qt window (mouthpiece.app.preview_window) only paints `PreviewState`s produced here,
so this module is fully testable without a camera or a display.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mouthpiece.config import Config
from mouthpiece.preprocess.landmarks import Landmarks
from mouthpiece.preprocess.mouth_roi import AlignedFrame, CropError
from mouthpiece.preprocess.pipeline import build_cropper, process_clip, save_clip
from mouthpiece.preprocess.smoothing import CausalSmoother

Frame = npt.NDArray[np.uint8]
DetectFn = Callable[[Frame, int], Landmarks | None]


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._t: deque[float] = deque()

    def tick(self, t: float | None = None) -> None:
        now = time.monotonic() if t is None else t
        self._t.append(now)
        while self._t and now - self._t[0] > self.window_s:
            self._t.popleft()

    @property
    def rate(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def mouth_openness(lm: Landmarks) -> float:
    """Inner-lip gap (iBUG 62↔66) divided by mouth width (48↔54). ~0 closed, ~0.5 wide open."""
    width = float(np.linalg.norm(lm[54] - lm[48]))
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[66] - lm[62])) / width


@dataclass(frozen=True)
class PreviewState:
    frame: Frame                     # BGR, unmirrored
    landmarks: Landmarks | None      # (smoothed) landmarks in frame pixels
    aligned: AlignedFrame | None
    camera_fps: float
    native_fps: float
    process_ms: float
    smoothing: bool
    recording_s: float | None        # None when not recording
    openness_history: tuple[float, ...]
    message: str | None = None       # one-shot notification (e.g. "saved clip")


class PreviewEngine:
    def __init__(
        self,
        cfg: Config,
        detect: DetectFn,
        native_fps: float = 0.0,
        out_root: Path | None = None,
        history: int = 100,
    ) -> None:
        self.cfg = cfg
        self.detect = detect
        self.native_fps = native_fps
        self.out_root = out_root or cfg.resolve("data/debug")
        self.cropper = build_cropper(cfg)
        self.smoother = CausalSmoother(cfg.preprocess.live_smoothing_window)
        self.smoothing = True
        self._meter = RateMeter()
        self._proc_ms: deque[float] = deque(maxlen=30)
        self._openness: deque[float] = deque([0.0] * history, maxlen=history)
        self._t0: float | None = None
        self._rec_frames: list[Frame] | None = None
        self._rec_landmarks: list[Landmarks | None] = []
        self._pending_message: str | None = None

    # --- commands ----------------------------------------------------------

    def toggle_smoothing(self) -> None:
        self.smoothing = not self.smoothing
        self.smoother.reset()

    @property
    def recording(self) -> bool:
        return self._rec_frames is not None

    def toggle_recording(self) -> None:
        if self._rec_frames is None:
            self.start_recording()
            return
        frames, lms = self.take_recording()
        self._pending_message = self._save(frames, lms)

    def start_recording(self) -> None:
        if self._rec_frames is None:
            self._rec_frames, self._rec_landmarks = [], []

    def take_recording(self) -> tuple[list[Frame], list[Landmarks | None]]:
        """Stop recording and hand over the raw frames and landmarks (push-to-talk)."""
        frames, lms = self._rec_frames or [], self._rec_landmarks
        self._rec_frames, self._rec_landmarks = None, []
        return frames, lms

    def notify(self, message: str) -> None:
        """Show a one-shot message with the next state."""
        self._pending_message = message

    def _save(self, frames: list[Frame], lms: list[Landmarks | None]) -> str:
        if len(frames) < 2:
            return "Clip too short — not saved"
        out = self.out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            result = process_clip(self.cfg, frames, self.cfg.capture.fps, lms)
        except CropError as exc:
            return f"Not saved: {exc}"
        save_clip(out, result, frames)
        return f"Saved {len(frames) / self.cfg.capture.fps:.1f}s clip → {out}"

    # --- per frame ---------------------------------------------------------

    def process(self, frame: Frame, t: float) -> PreviewState:
        self._t0 = t if self._t0 is None else self._t0
        self._meter.tick(t)
        start = time.perf_counter()
        raw = self.detect(frame, int(round((t - self._t0) * 1000)))
        lm = self.smoother.update(raw) if self.smoothing else raw
        aligned: AlignedFrame | None = None
        if lm is not None:
            try:
                aligned = self.cropper.align(frame, lm)
            except CropError:
                aligned = None
        self._proc_ms.append((time.perf_counter() - start) * 1000)
        self._openness.append(mouth_openness(lm) if lm is not None else 0.0)

        if self._rec_frames is not None:
            self._rec_frames.append(frame.copy())
            self._rec_landmarks.append(raw)

        message, self._pending_message = self._pending_message, None
        rec_s = (
            len(self._rec_frames) / self.cfg.capture.fps if self._rec_frames is not None else None
        )
        return PreviewState(
            frame=frame,
            landmarks=lm,
            aligned=aligned,
            camera_fps=self._meter.rate,
            native_fps=self.native_fps,
            process_ms=float(np.mean(self._proc_ms)),
            smoothing=self.smoothing,
            recording_s=rec_s,
            openness_history=tuple(self._openness),
            message=message,
        )
