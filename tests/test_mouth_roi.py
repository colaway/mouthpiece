import numpy as np
import pytest
from conftest import apply, similarity, synthetic_face

from mouthpiece.preprocess.mouth_roi import CropError, MouthCropper, cut_patch, to_model_input


@pytest.mark.parametrize(
    "scale,angle,tx,ty",
    [(1.0, 0, 190, 100), (1.6, 12, 120, 30), (0.8, -15, 230, 150), (1.2, 25, 200, 40)],
)
def test_alignment_recovers_mean_face(
    mean_face: np.ndarray, scale: float, angle: float, tx: float, ty: float
) -> None:
    m = similarity(scale, angle, tx, ty)
    frame, lm = synthetic_face(mean_face, m)
    cropper = MouthCropper(mean_face)
    out = cropper.align(frame, lm)
    assert out.face.shape == (256, 256)
    assert out.mouth.shape == (96, 96)
    np.testing.assert_allclose(out.landmarks, mean_face, atol=0.5)
    # The bright mouth polygon is centered in the crop.
    ys, xs = np.nonzero(out.mouth > 200)
    assert len(xs) > 100
    assert abs(xs.mean() - 48) < 4 and abs(ys.mean() - 48) < 4


def test_alignment_robust_to_small_noise(mean_face: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    m = similarity(1.3, 8, 170, 60)
    frame, lm = synthetic_face(mean_face, m)
    noisy = lm + rng.normal(0, 1.0, lm.shape).astype(np.float32)
    out = MouthCropper(mean_face).align(frame, noisy)
    center = out.landmarks[48:68].mean(0)
    assert np.linalg.norm(center - mean_face[48:68].mean(0)) < 3


def test_cut_patch_pads_at_border() -> None:
    img = np.full((256, 256), 7, dtype=np.uint8)
    pts = np.array([[100.0, 100.0]], dtype=np.float32)
    patch = cut_patch(img, pts, 48)
    assert patch.shape == (96, 96) and patch.min() == 7
    small = np.full((100, 100), 7, dtype=np.uint8)
    edge = cut_patch(small, np.array([[50.0, 90.0]], dtype=np.float32), 48)
    assert edge.shape == (96, 96)
    assert (edge[-38:] == 0).all()  # beyond the image bottom is zero-padded
    assert (edge[:-38] == 7).all()


def test_cut_patch_rejects_off_center() -> None:
    img = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(CropError):
        cut_patch(img, np.array([[250.0, 128.0]], dtype=np.float32), 48)


def test_crop_clip_handles_missing_frames(mean_face: np.ndarray) -> None:
    frames, lms = [], []
    for i in range(10):
        f, lm = synthetic_face(mean_face, similarity(1.2, i, 180 + 2 * i, 80))
        frames.append(f)
        lms.append(None if i in (0, 4, 5) else lm)
    mouths = MouthCropper(mean_face).crop_clip(frames, lms, window=12)
    assert mouths.shape == (10, 96, 96) and mouths.dtype == np.uint8
    with pytest.raises(CropError):
        MouthCropper(mean_face).crop_clip(frames, [None] * 10)
    with pytest.raises(ValueError):
        MouthCropper(mean_face).crop_clip(frames, lms[:5])


def test_to_model_input() -> None:
    mouths = np.full((5, 96, 96), 255, dtype=np.uint8)
    mouths[:, 4:92, 4:92] = 0
    x = to_model_input(mouths, 88, 0.421, 0.165)
    assert x.shape == (5, 88, 88) and x.dtype == np.float32
    np.testing.assert_allclose(x, (0 - 0.421) / 0.165, rtol=1e-6)  # border fully removed
    with pytest.raises(ValueError):
        to_model_input(mouths, 100)


def test_apply_helper_roundtrip(mean_face: np.ndarray) -> None:
    m = similarity(2.0, 30, 10, 20)
    pts = apply(m, mean_face)
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    np.testing.assert_allclose(apply(inv, pts), mean_face, atol=1e-3)
