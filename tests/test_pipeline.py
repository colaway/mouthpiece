import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import MEAN_FACE, similarity, synthetic_face

from mouthpiece.capture.camera import read_video
from mouthpiece.config import Config
from mouthpiece.preprocess.pipeline import ClipResult, process_clip, save_clip, write_video
from mouthpiece.preprocess.preview_engine import RateMeter


def test_process_clip_with_given_landmarks(cfg: Config, mean_face: np.ndarray) -> None:
    if not MEAN_FACE.exists():
        pytest.skip("mean face not downloaded")
    frames, lms = zip(*[synthetic_face(mean_face, similarity(1.1, 0, 200, 90)) for _ in range(8)],
                      strict=True)
    result = process_clip(cfg, list(frames), 25.0, list(lms))
    assert result.mouths.shape == (8, 96, 96)
    assert result.detection_rate == 1.0


def test_save_clip_and_read_back(tmp_path: Path) -> None:
    mouths = np.random.default_rng(0).integers(0, 255, (12, 96, 96), dtype=np.uint8)
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(12)]
    res = ClipResult(mouths, [None] * 6 + [np.zeros((68, 2), np.float32)] * 6, 25.0)
    out = save_clip(tmp_path / "clip", res, frames)
    np.testing.assert_array_equal(np.load(out / "mouth.npy"), mouths)
    meta = json.loads((out / "meta.json").read_text())
    assert meta == {"frames": 12, "fps": 25.0, "duration_s": 0.48, "detection_rate": 0.5}
    back, fps = read_video(str(out / "video.mp4"), target_fps=25.0)
    assert len(back) == 12 and fps == 25.0


def test_read_video_resamples_30_to_25(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    write_video(path, [np.full((64, 64, 3), i, np.uint8) for i in range(60)], 30.0)
    frames, fps = read_video(str(path), target_fps=25.0)
    assert fps == 25.0 and abs(len(frames) - 50) <= 1


def test_rate_meter() -> None:
    meter = RateMeter(window_s=10)
    for i in range(26):
        meter.tick(i / 25)
    assert abs(meter.rate - 25) < 0.01


def test_config_has_preprocess_assets(cfg: Config) -> None:
    assert cfg.preprocess.stable_points == (28, 33, 36, 39, 42, 45, 48, 54)
    assert replace(cfg.preprocess).live_smoothing_window > 0
    assert cv2.__version__
