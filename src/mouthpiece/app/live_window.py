"""`mouthpiece live`: the preview window plus push-to-talk transcription.

Hold Space while mouthing a phrase; on release the clip is cropped (exact offline
preprocessing, reusing the landmarks detected live) and decoded in a background thread.
"""

from __future__ import annotations

import queue
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout

from mouthpiece.app.preview_window import AMBER, PreviewWindow, mono_font, section
from mouthpiece.config import Config
from mouthpiece.decode.recognizer import Recognizer, Transcript


def format_transcript(t: Transcript, fps: float = 25.0) -> tuple[str, str]:
    """(n-best text, timing line) for the sidebar."""
    alts = "\n".join(f"{i}. {n.text or '∅'}  ({n.score:.1f})" for i, n in enumerate(t.nbest, 1))
    tm = t.timing
    timing = (
        f"{t.frames / fps:.1f}s clip · crop {tm.preprocess_s:.2f} · enc {tm.encoder_s:.2f} · "
        f"dec {tm.decode_s:.2f} · total {tm.total_s:.2f}s"
    )
    return alts, timing


class TranscribeWorker(QThread):
    """Loads the recognizer once, then transcribes clips from a queue.

    Queue items are (frames, landmarks, mode). "dictation" runs the free beam search;
    "command" ranks the closed set of `commands` phrases (needs `commands`).
    """

    status = Signal(str)
    ready = Signal(object)            # the loaded Recognizer (device, active weights)
    result = Signal(object, str)      # Transcript, mode
    failed = Signal(str)

    def __init__(
        self, cfg: Config, device: str | None = None, use_lm: bool = True,
        commands: Sequence[str] | None = None, hint: str = "hold [space] and mouth a phrase",
    ) -> None:
        super().__init__()
        self.cfg, self.device, self.use_lm = cfg, device, use_lm
        self.commands, self.hint = commands, hint
        self.clips: queue.Queue[tuple[Any, Any, str] | None] = queue.Queue()

    def stop(self) -> None:
        self.clips.put(None)
        self.wait(10000)

    def run(self) -> None:
        import time

        import numpy as np

        from mouthpiece.preprocess.mouth_roi import CropError
        from mouthpiece.preprocess.pipeline import process_clip

        try:
            self.status.emit("loading model…")
            rec = Recognizer.from_config(self.cfg, self.device, use_lm=self.use_lm)
            phrases = rec.phrase_set(self.commands) if self.commands else None
            warm = np.zeros((25, 96, 96), dtype=np.uint8)
            rec.transcribe(warm)  # warm up MPS kernels
            if phrases is not None:
                rec.transcribe_closed_set(warm, phrases)
            self.ready.emit(rec)
            self.status.emit(f"ready on {rec.device.type} · {self.hint}")
        except Exception as exc:
            self.failed.emit(f"model failed to load: {exc}")
            return
        while (item := self.clips.get()) is not None:
            frames, landmarks, mode = item
            if len(frames) < 10:
                self.status.emit(f"clip too short — {self.hint}")
                continue
            self.status.emit("transcribing…")
            try:
                t0 = time.perf_counter()
                clip = process_clip(self.cfg, frames, self.cfg.capture.fps, landmarks)
                pre = time.perf_counter() - t0
                if mode == "command" and phrases is not None:
                    t = rec.transcribe_closed_set(clip.mouths, phrases, pre)
                else:
                    t = rec.transcribe(clip.mouths, pre)
                self.result.emit(t, mode)
                self.status.emit(f"ready · {self.hint}")
            except CropError as exc:
                self.status.emit(f"no usable face in clip: {exc}")
            except Exception as exc:  # keep the app alive
                self.failed.emit(str(exc))


class LiveWindow(PreviewWindow):
    def __init__(
        self, cfg: Config, mirror: bool = True, device: str | None = None, use_lm: bool = True
    ) -> None:
        super().__init__(cfg, mirror)
        self.transcriber = TranscribeWorker(cfg, device, use_lm)
        self.setWindowTitle("mouthpiece — live")
        # The record button only mirrors push-to-talk state here.
        self.record_btn.clicked.disconnect()
        self.record_btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._held = False
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)  # catch Space whichever widget has focus

    def add_sections(self, side: QVBoxLayout) -> None:
        side.addSpacing(6)
        side.addLayout(section("transcript"))
        self.transcript_label = QLabel("—")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setFont(mono_font(15))
        self.transcript_label.setStyleSheet(f"color: {AMBER.name()};")
        self.nbest_label = QLabel("")
        self.nbest_label.setObjectName("stat")
        self.nbest_label.setWordWrap(True)
        self.timing_label = QLabel("")
        self.timing_label.setObjectName("meta")
        self.timing_label.setWordWrap(True)
        self.status_label = QLabel("starting…")
        self.status_label.setObjectName("meta")
        self.status_label.setWordWrap(True)
        for w in (self.transcript_label, self.nbest_label, self.timing_label, self.status_label):
            side.addWidget(w)

    def hint_text(self) -> str:
        a = AMBER.name()
        return (
            f"<span style='color:{a}'>[space]</span> hold to talk  "
            f"<span style='color:{a}'>[m]</span> mirror  "
            f"<span style='color:{a}'>[q]</span> quit"
        )

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and isinstance(
            event, QKeyEvent
        ) and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._push_to_talk(event.type() == QEvent.Type.KeyPress)
            return True
        return super().eventFilter(obj, event)

    def _push_to_talk(self, down: bool) -> None:
        if down == self._held:
            return
        self._held = down
        self._send("ptt_start" if down else "ptt_stop")

    def start(self) -> None:
        self.transcriber.status.connect(self.status_label.setText)
        self.transcriber.result.connect(lambda t, _mode: self.show_transcript(t))
        self.transcriber.failed.connect(lambda m: self.show_toast(m, error=True))
        self.transcriber.start()
        super().start()

    def on_clip(self, frames: object, landmarks: object) -> None:
        self.transcriber.clips.put((frames, landmarks, "dictation"))

    def show_transcript(self, t: Transcript) -> None:
        self.transcript_label.setText(t.text or "∅ (nothing recognised)")
        alts, timing = format_transcript(t, self.cfg.capture.fps)
        self.nbest_label.setText(alts)
        self.timing_label.setText(timing)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.transcriber.stop()
        super().closeEvent(event)


def run_live(cfg: Config, mirror: bool = True, device: str | None = None,
             use_lm: bool = True) -> None:
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    app.setApplicationName("Mouthpiece")
    app.setFont(mono_font(12))
    win = LiveWindow(cfg, mirror=mirror, device=device, use_lm=use_lm)
    win.show()
    win.start()
    app.exec()
