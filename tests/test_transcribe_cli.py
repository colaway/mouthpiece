from pathlib import Path

import numpy as np
import pytest

from mouthpiece.cli import main

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / p for p in ("models/vsr/model.pth", "models/vsr/model.json",
                             "models/vsr/unigram5000_units.txt")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="weights not downloaded")
def test_transcribe_npy_with_real_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clip = tmp_path / "clip"
    clip.mkdir()
    rng = np.random.default_rng(0)
    np.save(clip / "mouth.npy", rng.integers(0, 255, (15, 96, 96), dtype=np.uint8))
    code = main(["transcribe", str(clip), "--device", "cpu", "--no-lm", "--beam-size", "4",
                 "--top-k", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "transcript:" in out and "top-2:" in out
    for key in ("preprocess", "encoder", "decode", "total"):
        assert key in out


def test_transcribe_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["transcribe", str(tmp_path / "missing.mp4")]) == 1
    assert "Could not read" in capsys.readouterr().out
