import numpy as np
import pytest

from mouthpiece.capture.camera import FrameRateLimiter, resample_indices


def _accepted(src_fps: float, target: float, seconds: float = 4.0, jitter: float = 0.0) -> int:
    rng = np.random.default_rng(1)
    lim = FrameRateLimiter(target)
    ts = np.arange(0, seconds, 1 / src_fps) + rng.normal(0, jitter, int(seconds * src_fps) + 1)[
        : len(np.arange(0, seconds, 1 / src_fps))
    ]
    return sum(lim.accept(float(t)) for t in np.sort(ts))


@pytest.mark.parametrize("src", [30.0, 60.0])
def test_limiter_downsamples_to_target(src: float) -> None:
    n = _accepted(src, 25.0, seconds=4.0)
    assert abs(n / 4.0 - 25.0) <= 1.0


def test_limiter_with_jitter() -> None:
    n = _accepted(30.0, 25.0, seconds=4.0, jitter=0.002)
    assert abs(n / 4.0 - 25.0) <= 1.5


def test_limiter_passes_slower_stream() -> None:
    assert _accepted(15.0, 25.0, seconds=2.0) == 30
    assert _accepted(25.0, 25.0, seconds=2.0) == 50


def test_limiter_reanchors_after_gap() -> None:
    lim = FrameRateLimiter(25.0)
    assert lim.accept(0.0)
    assert lim.accept(10.0)  # long gap -> accept immediately
    assert not lim.accept(10.01)


def test_limiter_rejects_bad_fps() -> None:
    with pytest.raises(ValueError):
        FrameRateLimiter(0)


def test_resample_indices_30_to_25() -> None:
    ts = [i / 30 for i in range(90)]  # 3 s minus one frame
    idx = resample_indices(ts, 25.0)
    assert len(idx) == 75
    assert idx[0] == 0
    assert all(b >= a for a, b in zip(idx, idx[1:], strict=False))
    assert resample_indices([], 25.0) == []
