import numpy as np

from mouthpiece.preprocess.smoothing import CausalSmoother, interpolate_missing, smooth_centered


def _lm(v: float) -> np.ndarray:
    return np.full((68, 2), v, dtype=np.float32)


def test_interpolate_missing_linear_and_edges() -> None:
    out = interpolate_missing([None, _lm(0), None, None, _lm(3), None])
    assert out is not None
    assert [float(x[0, 0]) for x in out] == [0, 0, 1, 2, 3, 3]


def test_interpolate_all_missing() -> None:
    assert interpolate_missing([None, None]) is None


def test_smooth_centered_constant_is_identity() -> None:
    seq = [_lm(5)] * 10
    for s in smooth_centered(seq, 12):
        np.testing.assert_allclose(s, _lm(5))


def test_smooth_centered_preserves_centroid_and_edges() -> None:
    rng = np.random.default_rng(0)
    seq = [rng.normal(100, 3, (68, 2)).astype(np.float32) for _ in range(20)]
    out = smooth_centered(seq, 12)
    for a, b in zip(seq, out, strict=True):
        np.testing.assert_allclose(a.mean(0), b.mean(0), atol=1e-3)
    np.testing.assert_allclose(out[0], seq[0])  # window shrinks to 0 at the edges
    np.testing.assert_allclose(out[-1], seq[-1])
    # Shape jitter is reduced in the middle of the clip.
    jitter_in = np.std(np.stack(seq)[5:15] - np.stack(seq)[5:15].mean(1, keepdims=True), axis=0)
    jitter_out = np.std(np.stack(out)[5:15] - np.stack(out)[5:15].mean(1, keepdims=True), axis=0)
    assert jitter_out.mean() < 0.6 * jitter_in.mean()


def test_causal_smoother_reduces_jitter_and_resets() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 200, (68, 2)).astype(np.float32)
    sm = CausalSmoother(window=6, max_missing=2)
    raw, smoothed = [], []
    for _ in range(60):
        x = base + rng.normal(0, 2, base.shape).astype(np.float32)
        raw.append(x)
        out = sm.update(x)
        assert out is not None
        smoothed.append(out)
    err_raw = np.abs(np.stack(raw[10:]) - base).mean()
    err_sm = np.abs(np.stack(smoothed[10:]) - base).mean()
    assert err_sm < 0.7 * err_raw
    # Brief dropouts hold the last value; long ones reset.
    assert sm.update(None) is not None
    assert sm.update(None) is not None
    assert sm.update(None) is None
