"""Temporal landmark smoothing.

`interpolate_missing` + `smooth_centered` reproduce Auto-AVSR's offline preprocessing
exactly (used for clips fed to the model). `CausalSmoother` is a low-latency variant
for the live preview, where future frames are not available yet.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float32]


def interpolate_missing(landmarks: Sequence[Arr | None]) -> list[Arr] | None:
    """Linearly fill frames with no detection; copy nearest valid at the edges.

    Returns None if no frame has landmarks.
    """
    valid = [i for i, lm in enumerate(landmarks) if lm is not None]
    if not valid:
        return None
    out: list[Arr | None] = list(landmarks)
    for a, b in zip(valid[:-1], valid[1:], strict=True):
        if b - a > 1:
            la, lb = out[a], out[b]
            assert la is not None and lb is not None
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                out[k] = ((1 - w) * la + w * lb).astype(np.float32)
    first, last = valid[0], valid[-1]
    for k in range(first):
        out[k] = out[first]
    for k in range(last + 1, len(out)):
        out[k] = out[last]
    return [np.asarray(x, dtype=np.float32) for x in out if x is not None]


def smooth_centered(landmarks: Sequence[Arr], window: int = 12) -> list[Arr]:
    """Auto-AVSR smoothing: average over ±window//2 frames (shrinking at the edges),
    then shift so the centroid matches the frame's own landmarks."""
    n = len(landmarks)
    stack = np.stack(landmarks).astype(np.float32)
    out: list[Arr] = []
    for i in range(n):
        m = min(window // 2, i, n - 1 - i)
        smoothed = stack[i - m : i + m + 1].mean(axis=0)
        smoothed += stack[i].mean(axis=0) - smoothed.mean(axis=0)
        out.append(smoothed.astype(np.float32))
    return out


class CausalSmoother:
    """Moving average over the last `window` frames, centroid-locked to the current frame.

    Removes per-point jitter (shape noise) while keeping the face position responsive.
    Resets after `max_missing` consecutive frames without a face.
    """

    def __init__(self, window: int = 6, max_missing: int = 5) -> None:
        self.window = max(1, window)
        self.max_missing = max_missing
        self._buf: deque[Arr] = deque(maxlen=self.window)
        self._missing = 0

    def reset(self) -> None:
        self._buf.clear()
        self._missing = 0

    def update(self, lm: Arr | None) -> Arr | None:
        if lm is None:
            self._missing += 1
            if self._missing > self.max_missing:
                self.reset()
            return self._buf[-1].copy() if self._buf else None
        self._missing = 0
        self._buf.append(np.asarray(lm, dtype=np.float32))
        smoothed: Arr = np.mean(np.stack(self._buf), axis=0).astype(np.float32)
        smoothed += self._buf[-1].mean(axis=0) - smoothed.mean(axis=0)
        return smoothed
