"""`mouthpiece record`: collect your own mouthed clips with known transcripts.

A phrase is shown above the camera. Hold Space and mouth it; on release the take is
validated, cropped, and (optionally) transcribed by the current model in a background
thread. Then Enter keeps it, Backspace (or recording again) redoes it, N skips the phrase.
"""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from mouthpiece.app.preview_window import (
    AMBER,
    DIM,
    FG,
    REC,
    PreviewWindow,
    leader,
    mono_font,
    section,
)
from mouthpiece.config import Config
from mouthpiece.record.prompts import Prompt
from mouthpiece.record.session import Phase, RecorderSession, Take, process_take


class TakeWorker(QThread):
    """Validates/crops takes and (if enabled) transcribes them with the current model."""

    status = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: Config, baseline: bool, device: str | None = None,
                 use_lm: bool = True) -> None:
        super().__init__()
        self.cfg, self.baseline, self.device, self.use_lm = cfg, baseline, device, use_lm
        self.jobs: queue.Queue[tuple[Prompt, Any, Any] | None] = queue.Queue()

    def stop(self) -> None:
        self.jobs.put(None)
        self.wait(10000)

    def run(self) -> None:
        transcribe = None
        if self.baseline:
            try:
                import numpy as np

                from mouthpiece.decode.recognizer import Recognizer

                self.status.emit("loading model for baseline transcripts…")
                rec = Recognizer.from_config(self.cfg, self.device, use_lm=self.use_lm)
                rec.transcribe(np.zeros((25, 96, 96), dtype=np.uint8))  # warm up MPS
                transcribe = lambda m: rec.transcribe(m).text  # noqa: E731
                self.status.emit(f"model ready on {rec.device.type}")
            except Exception as exc:  # recording still works without a baseline
                self.failed.emit(f"baseline disabled, model failed to load: {exc}")
        while (job := self.jobs.get()) is not None:
            prompt, frames, landmarks = job
            try:
                self.result.emit(process_take(self.cfg, prompt, frames, landmarks, transcribe))
            except Exception as exc:  # keep the window alive
                self.result.emit(Take(prompt, list(frames), error=str(exc)))


class RecorderWindow(PreviewWindow):
    def __init__(self, cfg: Config, session: RecorderSession, mirror: bool = True,
                 device: str | None = None, use_lm: bool = True) -> None:
        self.session = session
        super().__init__(cfg, mirror)
        self.worker_takes = TakeWorker(cfg, cfg.record.baseline, device, use_lm)
        self.setWindowTitle("mouthpiece — record")
        self.record_btn.clicked.disconnect()
        self.record_btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._held = False
        self._build_banner()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.refresh()

    # --- layout --------------------------------------------------------------

    def _build_banner(self) -> None:
        """Prompt banner above the camera image, where the eyes already are."""
        central = self.centralWidget()
        outer = central.layout() if central is not None else None
        if not isinstance(outer, QHBoxLayout):
            return
        outer.removeWidget(self.video)
        col = QVBoxLayout()
        col.setSpacing(6)
        banner = QWidget()
        b = QVBoxLayout(banner)
        b.setContentsMargins(0, 0, 0, 8)
        b.setSpacing(4)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("meta")
        self.phrase_label = QLabel("")
        self.phrase_label.setWordWrap(True)
        self.phrase_label.setMinimumHeight(80)
        self.take_label = QLabel("")
        self.take_label.setWordWrap(True)
        self.take_label.setStyleSheet("font-size: 14px;")
        self.take_label.setTextFormat(Qt.TextFormat.RichText)
        for w in (self.progress_label, self.phrase_label, self.take_label):
            b.addWidget(w)
        col.addWidget(banner)
        col.addWidget(self.video, stretch=1)
        outer.insertLayout(0, col, stretch=1)

    def add_sections(self, side: QVBoxLayout) -> None:
        side.addSpacing(6)
        side.addLayout(section("session"))
        self.session_labels: dict[str, QLabel] = {}
        for key in ("kept", "rejected", "redone", "skipped", "baseline wer"):
            lab = QLabel(leader(key, "--"))
            lab.setObjectName("stat")
            side.addWidget(lab)
            self.session_labels[key] = lab
        self.status_label = QLabel("starting…")
        self.status_label.setObjectName("meta")
        self.status_label.setWordWrap(True)
        side.addWidget(self.status_label)

    def hint_text(self) -> str:
        a = AMBER.name()

        def k(s: str) -> str:
            return f"<span style='color:{a}'>[{s}]</span>"

        return (f"{k('space')} hold to record  {k('enter')} keep  {k('⌫')} redo<br>"
                f"{k('n')} skip phrase  {k('m')} mirror  {k('q')} quit")

    # --- state -> view -------------------------------------------------------

    def refresh(self) -> None:
        s = self.session
        st = s.stats
        p = s.prompt
        if p is None:
            self.progress_label.setText(f"session complete · {st.kept} kept")
            self.phrase_label.setText("done — thanks!")
            self.phrase_label.setStyleSheet(f"color: {FG.name()}; font-size: 30px;")
        else:
            self.progress_label.setText(
                f"take {st.kept + 1} / {s.target} · {p.kind} · "
                f"{s.takes_of(p)} of {s.scheduler.takes_per_phrase} takes of this phrase"
            )
            self.phrase_label.setText(p.text)
            color = REC if s.phase is Phase.RECORDING else AMBER
            self.phrase_label.setStyleSheet(f"color: {color.name()}; font-size: 30px;")
        self.take_label.setText(self._take_text())
        self.session_labels["kept"].setText(leader("kept", f"{st.kept} / {s.target}"))
        for key in ("rejected", "redone", "skipped"):
            self.session_labels[key].setText(leader(key, str(getattr(st, key))))
        bw = st.baseline_wer
        self.session_labels["baseline wer"].setText(
            leader("baseline wer", "--" if bw is None else f"{bw:.0%}"))

    def _take_text(self) -> str:
        s = self.session
        dim = DIM.name()
        msg = f"<span style='color:{dim}'>{s.message}</span><br>" if s.message else ""
        if s.phase is Phase.PROMPT:
            return msg + "hold <b>[space]</b> and mouth the phrase silently"
        if s.phase is Phase.RECORDING:
            return f"<span style='color:{REC.name()}'>● recording — release when done</span>"
        if s.phase is Phase.PROCESSING:
            return "checking take…"
        if s.phase is Phase.DONE:
            return msg + "close the window (q) — run `mouthpiece dataset stats` for a summary"
        take = s.take
        if take is None or take.clip is None:
            return ""
        dur = take.clip.mouths.shape[0] / take.clip.fps
        head = f"{dur:.1f}s · face {take.clip.detection_rate:.0%}"
        if take.hypothesis is not None and take.wer is not None:
            head += (f" · model heard: <span style='color:{FG.name()}'>"
                     f"{take.hypothesis or '∅'}</span> (WER {take.wer:.0%})")
        return head + "<br><b>[enter]</b> keep · <b>[⌫]</b> redo · <b>[n]</b> skip"

    # --- input ---------------------------------------------------------------

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) or not isinstance(
            event, QKeyEvent
        ):
            return super().eventFilter(obj, event)
        key, press = event.key(), event.type() == QEvent.Type.KeyPress
        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._push_to_talk(press)
            return True
        actions: dict[int, Callable[[], object]] = {
            Qt.Key.Key_Return.value: self.session.keep,
            Qt.Key.Key_Enter.value: self.session.keep,
            Qt.Key.Key_Backspace.value: self.session.redo,
            Qt.Key.Key_N.value: self.session.skip,
        }
        if key in actions:
            if press and not event.isAutoRepeat():
                actions[key]()
                self.refresh()
            return True
        return super().eventFilter(obj, event)

    def _push_to_talk(self, down: bool) -> None:
        if down == self._held:
            return
        if down and not self.session.begin_take():
            return
        self._held = down
        self._send("ptt_start" if down else "ptt_stop")
        self.refresh()

    def on_clip(self, frames: object, landmarks: object) -> None:
        prompt = self.session.end_take()
        if prompt is not None:
            self.worker_takes.jobs.put((prompt, frames, landmarks))
        self.refresh()

    def on_take(self, take: Take) -> None:
        self.session.submit(take)
        self.refresh()

    def start(self) -> None:
        self.worker_takes.status.connect(self.status_label.setText)
        self.worker_takes.result.connect(self.on_take)
        self.worker_takes.failed.connect(lambda m: self.show_toast(m, error=True))
        self.worker_takes.start()
        super().start()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.worker_takes.stop()
        super().closeEvent(event)


def run_record(cfg: Config, session: RecorderSession, mirror: bool = True,
               device: str | None = None, use_lm: bool = True) -> None:
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    app.setApplicationName("Mouthpiece")
    app.setFont(mono_font(12))
    win = RecorderWindow(cfg, session, mirror=mirror, device=device, use_lm=use_lm)
    win.show()
    win.start()
    app.exec()
