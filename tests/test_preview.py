import os
from pathlib import Path

import numpy as np
import pytest
from conftest import MEAN_FACE, similarity, synthetic_face

from mouthpiece.config import Config
from mouthpiece.preprocess.preview_engine import PreviewEngine, mouth_openness

pytestmark = pytest.mark.skipif(not MEAN_FACE.exists(), reason="mean face not downloaded")


def _engine(cfg: Config, tmp_path: Path, lm: np.ndarray | None) -> PreviewEngine:
    return PreviewEngine(cfg, lambda f, t: lm, native_fps=30, out_root=tmp_path)


def test_engine_produces_state(cfg: Config, mean_face: np.ndarray, tmp_path: Path) -> None:
    frame, lm = synthetic_face(mean_face, similarity(1.2, 5, 180, 60))
    eng = _engine(cfg, tmp_path, lm)
    states = [eng.process(frame, i / 25) for i in range(30)]
    s = states[-1]
    assert s.landmarks is not None and s.aligned is not None
    assert s.aligned.mouth.shape == (96, 96)
    assert abs(s.camera_fps - 25) < 0.5
    assert s.recording_s is None and s.smoothing
    assert len(s.openness_history) == 100


def test_engine_no_face(cfg: Config, mean_face: np.ndarray, tmp_path: Path) -> None:
    frame, _ = synthetic_face(mean_face, similarity(1.0, 0, 190, 100))
    s = _engine(cfg, tmp_path, None).process(frame, 0.0)
    assert s.landmarks is None and s.aligned is None


def test_engine_recording_saves_clip(cfg: Config, mean_face: np.ndarray, tmp_path: Path) -> None:
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    eng = _engine(cfg, tmp_path, lm)
    eng.toggle_recording()
    assert eng.recording
    for i in range(10):
        s = eng.process(frame, i / 25)
    assert s.recording_s == pytest.approx(0.4)
    eng.toggle_recording()
    s = eng.process(frame, 0.5)
    assert s.message is not None and s.message.startswith("Saved")
    clips = list(tmp_path.iterdir())
    assert len(clips) == 1 and (clips[0] / "mouth.npy").exists()
    assert eng.process(frame, 0.54).message is None  # one-shot


def test_engine_short_clip_not_saved(cfg: Config, mean_face: np.ndarray, tmp_path: Path) -> None:
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    eng = _engine(cfg, tmp_path, lm)
    eng.toggle_recording()
    eng.toggle_recording()
    assert "too short" in (eng.process(frame, 0).message or "")


def test_toggle_smoothing(cfg: Config, tmp_path: Path) -> None:
    eng = _engine(cfg, tmp_path, None)
    eng.toggle_smoothing()
    assert not eng.smoothing


def test_mouth_openness(mean_face: np.ndarray) -> None:
    closed = mean_face.copy()
    closed[66] = closed[62]
    assert mouth_openness(closed) == pytest.approx(0.0)
    opened = closed.copy()
    opened[66] = opened[62] + [0, 20]
    assert mouth_openness(opened) > 0.2
    assert mouth_openness(np.zeros((68, 2), np.float32)) == 0.0


def test_window_renders_offscreen(cfg: Config, mean_face: np.ndarray, tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from mouthpiece.app.preview_window import (
        PreviewWindow,
        bgr_to_qimage,
        contain_rect,
        leader,
    )

    app = QApplication.instance() or QApplication([])
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    win = PreviewWindow(cfg)
    for detected in (lm, None):
        eng = _engine(cfg, tmp_path, detected)
        eng.toggle_recording()
        win.update_state(eng.process(frame, 0.0))
        win.show()
        app.processEvents()
        img = win.grab().toImage()
        assert img.width() > 0
    win.show_error("camera unavailable")
    win.close()

    assert len(leader("camera", "25.0 fps")) == 22
    q = bgr_to_qimage(np.zeros((10, 20, 3), np.uint8))
    assert (q.width(), q.height()) == (20, 10)
    from PySide6.QtCore import QRectF

    r, scale = contain_rect(640, 480, QRectF(0, 0, 800, 800))
    assert scale == pytest.approx(1.25) and r.height() == pytest.approx(600)


def test_engine_push_to_talk_hands_over_clip(
    cfg: Config, mean_face: np.ndarray, tmp_path: Path
) -> None:
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    eng = _engine(cfg, tmp_path, lm)
    eng.start_recording()
    eng.start_recording()  # idempotent (key repeat)
    for i in range(5):
        eng.process(frame, i / 25)
    frames, lms = eng.take_recording()
    assert len(frames) == 5 and len(lms) == 5 and not eng.recording
    assert list(tmp_path.iterdir()) == []  # nothing saved to disk
    assert eng.take_recording() == ([], [])


def test_live_window_push_to_talk_offscreen(
    cfg: Config, mean_face: np.ndarray, tmp_path: Path
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    from mouthpiece.app.live_window import LiveWindow
    from mouthpiece.decode.recognizer import NBestEntry, Timing, Transcript

    app = QApplication.instance() or QApplication([])
    win = LiveWindow(cfg)
    sent: list[str] = []
    win._send = sent.append  # type: ignore[method-assign]
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier,
                       "", True)
    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    for ev in (press, repeat, release):
        assert win.eventFilter(win, ev)
    assert sent == ["ptt_start", "ptt_stop"]

    t = Transcript("HELLO", [NBestEntry("HELLO", -1.0, {}), NBestEntry("YELLOW", -2.0, {})],
                   frames=50, device="mps", timing=Timing(0.1, 0.07, 0.5))
    win.show_transcript(t)
    assert win.transcript_label.text() == "HELLO"
    assert "2. YELLOW" in win.nbest_label.text()
    assert "2.0s clip" in win.timing_label.text()
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    win.update_state(_engine(cfg, tmp_path, lm).process(frame, 0.0))
    win.show()
    app.processEvents()
    assert win.grab().toImage().width() > 0
    win.close()
