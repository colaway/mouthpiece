"""UI-independent recorder logic: validate a take, then keep / redo / skip it.

The window (mouthpiece.app.recorder_window) drives a `RecorderSession` from key presses and
a background worker turns raw push-to-talk frames into a `Take` with `process_take`, so all
of this is testable without a camera, display, or model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from mouthpiece.config import Config
from mouthpiece.metrics import corpus_wer, wer
from mouthpiece.preprocess.landmarks import Landmarks
from mouthpiece.preprocess.mouth_roi import CropError
from mouthpiece.preprocess.pipeline import ClipResult, Frame, process_clip
from mouthpiece.record.manifest import ManifestEntry, PersonalStore
from mouthpiece.record.prompts import Prompt, PromptScheduler

TranscribeFn = Callable[[npt.NDArray[np.uint8]], str]
EncodeFn = Callable[[str], list[int]]


@dataclass(frozen=True)
class Take:
    prompt: Prompt
    frames: list[Frame]
    clip: ClipResult | None = None
    hypothesis: str | None = None
    error: str | None = None          # why the take was rejected

    @property
    def ok(self) -> bool:
        return self.error is None and self.clip is not None

    @property
    def wer(self) -> float | None:
        return None if self.hypothesis is None else wer(self.prompt.text, self.hypothesis)


def check_take(cfg: Config, frames: Sequence[Frame],
               landmarks: Sequence[Landmarks | None]) -> str | None:
    """Rejection reason for a raw take, or None if it is usable."""
    r, fps = cfg.record, cfg.capture.fps
    n = len(frames)
    if n < r.min_frames:
        return f"too short ({n / fps:.1f}s, need ≥ {r.min_frames / fps:.1f}s)"
    if n > r.max_frames:
        return f"too long ({n / fps:.1f}s, max {r.max_frames / fps:.1f}s)"
    rate = sum(lm is not None for lm in landmarks) / max(len(landmarks), 1)
    if rate < r.min_detection_rate:
        return f"face found in only {rate:.0%} of frames (need {r.min_detection_rate:.0%})"
    return None


def process_take(
    cfg: Config,
    prompt: Prompt,
    frames: Sequence[Frame],
    landmarks: Sequence[Landmarks | None],
    transcribe: TranscribeFn | None = None,
) -> Take:
    """Validate, crop (exact offline preprocessing), and optionally transcribe a take."""
    frames = list(frames)
    reason = check_take(cfg, frames, landmarks)
    if reason is not None:
        return Take(prompt, frames, error=reason)
    try:
        clip = process_clip(cfg, frames, cfg.capture.fps, landmarks)
    except CropError as exc:
        return Take(prompt, frames, error=f"no usable face: {exc}")
    hyp = transcribe(clip.mouths) if transcribe is not None else None
    return Take(prompt, frames, clip, hyp)


class Phase(StrEnum):
    PROMPT = "prompt"           # showing a phrase, waiting for push-to-talk
    RECORDING = "recording"
    PROCESSING = "processing"   # take handed to the worker
    REVIEW = "review"           # valid take shown, waiting for keep / redo / skip
    DONE = "done"


@dataclass
class SessionStats:
    kept: int = 0
    rejected: int = 0
    redone: int = 0
    skipped: int = 0
    pairs: list[tuple[str, str]] = field(default_factory=list)   # (ref, baseline hyp)

    @property
    def baseline_wer(self) -> float | None:
        return corpus_wer(self.pairs) if self.pairs else None


class RecorderSession:
    def __init__(
        self,
        scheduler: PromptScheduler,
        store: PersonalStore,
        encode: EncodeFn,
        target: int = 100,
    ) -> None:
        self.scheduler, self.store, self.encode, self.target = scheduler, store, encode, target
        self.stats = SessionStats()
        self.take: Take | None = None
        self.message: str | None = None
        self.last_saved: ManifestEntry | None = None
        self.prompt: Prompt | None = None
        self.phase = Phase.PROMPT
        self._advance()

    # --- push-to-talk ----------------------------------------------------

    def begin_take(self) -> bool:
        """Start recording. Recording over a take under review discards it (redo)."""
        if self.phase is Phase.REVIEW:
            self.stats.redone += 1
            self.take = None
        elif self.phase is not Phase.PROMPT:
            return False
        self.phase, self.message = Phase.RECORDING, None
        return True

    def end_take(self) -> Prompt | None:
        """Stop recording; returns the prompt the take belongs to (None if not recording)."""
        if self.phase is not Phase.RECORDING:
            return None
        self.phase = Phase.PROCESSING
        return self.prompt

    def submit(self, take: Take) -> None:
        """Result from the worker."""
        if self.phase is not Phase.PROCESSING or take.prompt != self.prompt:
            return  # stale result
        if not take.ok:
            self.stats.rejected += 1
            self.phase, self.message = Phase.PROMPT, f"rejected: {take.error}"
            return
        self.take, self.phase, self.message = take, Phase.REVIEW, None

    # --- review ------------------------------------------------------------

    def keep(self) -> ManifestEntry | None:
        take = self.take
        if self.phase is not Phase.REVIEW or take is None or take.clip is None:
            return None
        entry = self.store.save(
            take.prompt.text, take.prompt.kind, self.encode(take.prompt.text), take.clip,
            take.frames, take.hypothesis, take.wer,
        )
        self.scheduler.record(take.prompt)
        self.stats.kept += 1
        if take.hypothesis is not None:
            self.stats.pairs.append((take.prompt.text, take.hypothesis))
        self.last_saved = entry
        self.message = f"saved {entry.id} ({entry.duration_s:.1f}s, {entry.split})"
        self._advance()
        return entry

    def redo(self) -> None:
        if self.phase is Phase.REVIEW:
            self.stats.redone += 1
            self.take, self.phase, self.message = None, Phase.PROMPT, "take discarded"

    def skip(self) -> None:
        if self.phase in (Phase.PROMPT, Phase.REVIEW) and self.prompt is not None:
            self.scheduler.skip(self.prompt)
            self.stats.skipped += 1
            self.message = f"skipped “{self.prompt.text}”"
            self._advance()

    # --- helpers -----------------------------------------------------------

    def takes_of(self, prompt: Prompt) -> int:
        return self.scheduler.counts[prompt.key]

    def _advance(self) -> None:
        self.take = None
        nxt = None if self.stats.kept >= self.target else self.scheduler.next()
        self.prompt = nxt
        self.phase = Phase.PROMPT if nxt is not None else Phase.DONE
