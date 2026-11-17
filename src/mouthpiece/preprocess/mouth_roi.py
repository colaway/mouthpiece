"""Face alignment and mouth ROI cropping, matching Auto-AVSR's preprocessing.

Pipeline per frame (from auto_avsr/preparation/detectors/retinaface/video_process.py):
  1. grayscale
  2. similarity transform (cv2.estimateAffinePartial2D, LMEDS) mapping the frame's
     stable points (iBUG 28, 33, 36, 39, 42, 45, 48, 54) onto the same points of the
     256x256 mean face
  3. warp to 256x256 (bilinear, black border)
  4. cut a 96x96 patch centered on the mean of the transformed mouth points (48-67)
Then for the model: center-crop 88x88, scale to [0, 1], normalize (0.421, 0.165).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mouthpiece.preprocess.landmarks import MOUTH
from mouthpiece.preprocess.smoothing import interpolate_missing, smooth_centered

Gray = npt.NDArray[np.uint8]
Arr = npt.NDArray[np.float32]


class CropError(RuntimeError):
    """The face is too far off-center/large after alignment to cut a mouth patch."""


@dataclass(frozen=True)
class AlignedFrame:
    face: Gray            # aligned_size x aligned_size grayscale face
    landmarks: Arr        # (68, 2) landmarks in aligned-face coordinates
    mouth: Gray           # roi_size x roi_size crop
    transform: npt.NDArray[np.float64]  # 2x3 affine, frame -> aligned face


def load_mean_face(path: str | Path) -> Arr:
    ref: Arr = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float32)
    if ref.shape != (68, 2):
        raise ValueError(f"mean face must be (68, 2), got {ref.shape}")
    return ref


def to_gray(frame: npt.NDArray[np.uint8]) -> Gray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[return-value]


def cut_patch(img: Gray, points: Arr, half: int, threshold: float = 5.0) -> Gray:
    """Cut a (2*half)^2 patch centered on mean(points); pad with zeros at borders."""
    cx, cy = points.mean(axis=0)
    h, w = img.shape[:2]
    if abs(cy - h / 2) > half + threshold or abs(cx - w / 2) > half + threshold:
        raise CropError("mouth too far from the aligned-face center")
    y0, x0 = int(round(cy - half)), int(round(cx - half))
    size = 2 * half
    out = np.zeros((size, size), dtype=img.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, h), min(x0 + size, w)
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


class MouthCropper:
    def __init__(
        self,
        mean_face: Arr,
        stable_points: Sequence[int] = (28, 33, 36, 39, 42, 45, 48, 54),
        aligned_size: int = 256,
        roi_size: int = 96,
        reference_size: int = 256,
    ) -> None:
        self.stable_points = list(stable_points)
        self.aligned_size = aligned_size
        self.roi_size = roi_size
        offset = (reference_size - aligned_size) / 2.0
        self.reference = mean_face[self.stable_points] - offset

    def estimate_transform(self, landmarks: Arr) -> npt.NDArray[np.float64]:
        src = landmarks[self.stable_points].astype(np.float32)
        m, _ = cv2.estimateAffinePartial2D(src, self.reference, method=cv2.LMEDS)
        if m is None:
            raise CropError("could not estimate face alignment")
        return m.astype(np.float64)

    def align(self, frame: npt.NDArray[np.uint8], landmarks: Arr) -> AlignedFrame:
        gray = to_gray(frame)
        m = self.estimate_transform(landmarks)
        s = self.aligned_size
        face: Gray = np.asarray(
            cv2.warpAffine(
                gray, m, (s, s), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ),
            dtype=np.uint8,
        )
        pts = (landmarks @ m[:, :2].T + m[:, 2]).astype(np.float32)
        mouth = cut_patch(face, pts[MOUTH], self.roi_size // 2)
        return AlignedFrame(face, pts, mouth, m)

    def crop_clip(
        self,
        frames: Sequence[npt.NDArray[np.uint8]],
        landmarks: Sequence[Arr | None],
        window: int = 12,
    ) -> Gray:
        """Offline (exact Auto-AVSR) processing of a whole clip -> (T, roi, roi) uint8."""
        if len(frames) != len(landmarks):
            raise ValueError("frames and landmarks must have the same length")
        filled = interpolate_missing(landmarks)
        if filled is None:
            raise CropError("no face detected in any frame")
        smoothed = smooth_centered(filled, window)
        return np.stack([self.align(f, lm).mouth for f, lm in zip(frames, smoothed, strict=True)])


def to_model_input(
    mouths: Gray, size: int = 88, mean: float = 0.421, std: float = 0.165
) -> npt.NDArray[np.float32]:
    """(T, 96, 96) uint8 -> (T, 88, 88) float32, center-cropped and normalized."""
    t, h, w = mouths.shape
    if size > h or size > w:
        raise ValueError("crop size larger than input")
    y0, x0 = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    x = mouths[:, y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255.0
    return ((x - mean) / std).astype(np.float32)
