from pathlib import Path

import numpy as np
import pytest

from mouthpiece.preprocess.landmarks import MOUTH, MP_TO_IBUG68, FaceLandmarker, mesh_to_ibug68

TASK = Path(__file__).resolve().parents[1] / "models/preprocess/face_landmarker.task"


def test_mapping_is_valid() -> None:
    assert len(MP_TO_IBUG68) == 68
    assert len(set(MP_TO_IBUG68)) == 68
    assert all(0 <= i < 468 for i in MP_TO_IBUG68)
    assert MOUTH == slice(48, 68)


def test_mesh_to_ibug68() -> None:
    mesh = np.arange(478 * 2, dtype=np.float32).reshape(478, 2)
    lm = mesh_to_ibug68(mesh)
    assert lm.shape == (68, 2)
    assert lm[36].tolist() == mesh[33].tolist()  # right eye outer corner
    with pytest.raises(ValueError):
        mesh_to_ibug68(np.zeros((10, 2)))


@pytest.mark.skipif(not TASK.exists(), reason="face_landmarker.task not downloaded")
def test_landmarker_no_face_returns_none() -> None:
    with FaceLandmarker(TASK) as lmk:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert lmk.detect(blank, 0) is None
        assert lmk.detect(blank, 0) is None  # repeated timestamp handled


def test_landmarker_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FaceLandmarker(tmp_path / "nope.task")
