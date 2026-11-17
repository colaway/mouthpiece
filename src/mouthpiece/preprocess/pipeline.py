"""Glue: frames -> landmarks -> mouth crops, plus saving clips to disk."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mouthpiece.config import Config
from mouthpiece.preprocess.landmarks import FaceLandmarker, Landmarks
from mouthpiece.preprocess.mouth_roi import MouthCropper, load_mean_face

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ClipResult:
    mouths: npt.NDArray[np.uint8]   # (T, 96, 96)
    landmarks: list[Landmarks | None]
    fps: float

    @property
    def detection_rate(self) -> float:
        if not self.landmarks:
            return 0.0
        return sum(lm is not None for lm in self.landmarks) / len(self.landmarks)


def build_cropper(cfg: Config) -> MouthCropper:
    p = cfg.preprocess
    return MouthCropper(
        load_mean_face(cfg.resolve(p.mean_face)), p.stable_points, p.aligned_size, p.roi_size
    )


def build_landmarker(cfg: Config) -> FaceLandmarker:
    return FaceLandmarker(cfg.resolve(cfg.preprocess.face_landmarker_model))


def detect_clip(
    landmarker: FaceLandmarker, frames: Sequence[Frame], fps: float
) -> list[Landmarks | None]:
    return [landmarker.detect(f, int(round(i * 1000.0 / fps))) for i, f in enumerate(frames)]


def process_clip(
    cfg: Config,
    frames: Sequence[Frame],
    fps: float,
    landmarks: Sequence[Landmarks | None] | None = None,
) -> ClipResult:
    """Exact (offline) Auto-AVSR preprocessing of a clip. Reuses landmarks if given."""
    if landmarks is None:
        with build_landmarker(cfg) as lmk:
            landmarks = detect_clip(lmk, frames, fps)
    mouths = build_cropper(cfg).crop_clip(frames, landmarks, cfg.preprocess.smoothing_window)
    return ClipResult(mouths, list(landmarks), fps)


def write_video(path: Path, frames: Sequence[npt.NDArray[np.uint8]], fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    is_color = frames[0].ndim == 3
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h), is_color)
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def save_clip(out_dir: Path, result: ClipResult, frames: Sequence[Frame] | None = None) -> Path:
    """Write mouth.npy (uint8 T x 96 x 96), mouth.mp4 preview, optional video.mp4, meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mouth.npy", result.mouths)
    write_video(out_dir / "mouth.mp4", list(result.mouths), result.fps)
    if frames:
        write_video(out_dir / "video.mp4", frames, result.fps)
    meta = {
        "frames": int(result.mouths.shape[0]),
        "fps": result.fps,
        "duration_s": round(result.mouths.shape[0] / result.fps, 3),
        "detection_rate": round(result.detection_rate, 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
