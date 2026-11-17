"""Face landmarks via MediaPipe Face Landmarker, mapped to the dlib/iBUG 68-point layout.

The pretrained Auto-AVSR weights were trained on mouth crops produced from 68-point
landmarks (RetinaFace + FAN). We run MediaPipe's 478-point mesh instead (fast, on-device)
and pick the mesh vertices that correspond to the 68 iBUG points, so the rest of the
pipeline (alignment to the mean face, crop around the mouth) matches training exactly.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

Landmarks = npt.NDArray[np.float32]  # (68, 2) pixel coordinates (x, y)

# MediaPipe 478-mesh vertex for each iBUG-68 point. "Right"/"left" are the subject's,
# i.e. iBUG point 36 (subject's right eye, outer corner) appears on the image's left.
MP_TO_IBUG68: tuple[int, ...] = (
    # jaw 0-16
    127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
    # right brow 17-21, left brow 22-26
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # nose bridge 27-30, nose base 31-35
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # right eye 36-41, left eye 42-47
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # inner lip 60-67
    78, 82, 13, 312, 308, 317, 14, 87,
)
assert len(MP_TO_IBUG68) == 68

MOUTH = slice(48, 68)


def mesh_to_ibug68(mesh_xy: npt.NDArray[np.floating]) -> Landmarks:
    """Select the 68 iBUG points from a (478, 2) MediaPipe mesh."""
    if mesh_xy.ndim != 2 or mesh_xy.shape[0] < 468:
        raise ValueError(f"expected (478, 2) mesh, got {mesh_xy.shape}")
    return np.asarray(mesh_xy[list(MP_TO_IBUG68), :2], dtype=np.float32)


class FaceLandmarker:
    """Thin wrapper over mediapipe.tasks FaceLandmarker in VIDEO mode.

    Call `detect(frame_bgr, timestamp_ms)` with strictly increasing timestamps; it
    returns (68, 2) pixel landmarks for the largest face, or None.
    """

    def __init__(self, model_path: str | Path, num_faces: int = 1) -> None:
        os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe/TFLite info logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Face landmarker model missing: {model_path}")
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: npt.NDArray[np.uint8], timestamp_ms: int) -> Landmarks | None:
        import cv2

        ts = max(int(timestamp_ms), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        best: Landmarks | None = None
        best_area = -1.0
        for face in result.face_landmarks:
            mesh = np.array([(p.x * w, p.y * h) for p in face], dtype=np.float32)
            lm = mesh_to_ibug68(mesh)
            span = lm.max(axis=0) - lm.min(axis=0)
            area = float(span[0] * span[1])
            if area > best_area:
                best, best_area = lm, area
        return best

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarker:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
