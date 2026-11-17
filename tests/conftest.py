from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mouthpiece.config import Config, load_config

MEAN_FACE = Path(__file__).resolve().parents[1] / "models/preprocess/20words_mean_face.npy"


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def mean_face() -> np.ndarray:
    if MEAN_FACE.exists():
        return np.load(MEAN_FACE).astype(np.float32)
    # Fallback: a plausible synthetic 68-point face so tests run without downloads.
    rng = np.random.default_rng(0)
    pts = rng.uniform(80, 180, size=(68, 2)).astype(np.float32)
    pts[36:42] = [95, 95]; pts[42:48] = [165, 95]  # noqa: E702
    pts[31:36] = [130, 135]; pts[48:68] = [130, 158]  # noqa: E702
    pts[48:68] += rng.normal(0, 8, size=(20, 2))
    return pts


def similarity(scale: float, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def synthetic_face(
    mean_face: np.ndarray, m: np.ndarray, size: tuple[int, int] = (480, 640)
) -> tuple[np.ndarray, np.ndarray]:
    """BGR frame with a bright filled mouth polygon, plus its 68 landmarks."""
    lm = apply(m, mean_face)
    img = np.full((*size, 3), 60, dtype=np.uint8)
    cv2.fillPoly(img, [lm[48:60].astype(np.int32)], (255, 255, 255))
    for p in lm[36:48]:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 0, 0), -1)
    return img, lm
