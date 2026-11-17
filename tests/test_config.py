from pathlib import Path

import pytest

from mouthpiece.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.capture.fps == 25
    assert cfg.preprocess.roi_size == 96
    assert cfg.preprocess.model_input_size == 88
    assert cfg.model.device in ("auto", "mps", "cpu")


def test_resolve_relative_to_root() -> None:
    cfg = load_config()
    assert cfg.resolve("models/x.pth") == cfg.root / "models/x.pth"
    assert cfg.resolve("/abs/x.pth") == Path("/abs/x.pth")


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("capture:\n  fsp: 30\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(p)


def test_unknown_section_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("captur:\n  fps: 30\n")
    with pytest.raises(ConfigError, match="Unknown config sections"):
        load_config(p)


def test_bad_device_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  device: cuda\n")
    with pytest.raises(ConfigError, match="device"):
        load_config(p)


def test_missing_sections_use_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("capture:\n  fps: 30\n")
    cfg = load_config(p)
    assert cfg.capture.fps == 30
    assert cfg.decode.beam_size == 40


def test_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("capture:\n  camera_index: 2\n")
    monkeypatch.setenv("MOUTHPIECE_CONFIG", str(p))
    assert load_config().capture.camera_index == 2
