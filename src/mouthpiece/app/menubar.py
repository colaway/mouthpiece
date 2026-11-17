"""`mouthpiece app`: the menu-bar app.

Lives in the menu bar (no Dock icon). While armed the camera runs; hold the hotkey (right ⌥
by default) while mouthing, release to transcribe. A popup near the cursor shows the top 3;
#1 is accepted after `app.picker_timeout_s` unless you press 1/2/3, Return, Esc or click.
In dictation mode the picker is skipped unless `app.dictation_picker` is on: #1 is typed at once.
Each accepted utterance is emitted as a `Commit` on `MenuBarApp.committed`; the typing stage
(`mouthpiece.typing`) types dictation into the focused app and turns command phrases into
keys and actions. Typing needs Accessibility permission.

Command mode ranks the closed set of command phrases; dictation mode decodes free text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from mouthpiece.app import permissions
from mouthpiece.app.controller import (
    MODES,
    Commit,
    Mode,
    PickerModel,
    PickerResult,
    alternatives,
    load_command_phrases,
    mode_switch,
    uses_picker,
)
from mouthpiece.app.hotkey import GlobalHotkey, KeyEvent, PttAction, PushToTalk, get_hotkey
from mouthpiece.app.live_window import TranscribeWorker
from mouthpiece.app.picker import CursorPopup
from mouthpiece.app.preview_window import AMBER, DIM, REC, CaptureWorker, mono_font
from mouthpiece.config import Config
from mouthpiece.decode.recognizer import Recognizer, Transcript
from mouthpiece.model.checkpoint import resolve_checkpoint
from mouthpiece.typing.actions import AppSignal
from mouthpiece.typing.emitter import QuartzEmitter
from mouthpiece.typing.output import TypingOutput

ICON_STATES = ("idle", "loading", "recording", "busy", "paused", "error")


def tray_icon(state: str) -> QIcon:
    """Mouth glyph for the menu bar. Idle/paused are template images (macOS tints them for
    light/dark menu bars); recording is red, busy/loading amber, error red with a slash."""
    if state not in ICON_STATES:
        raise ValueError(f"unknown icon state {state!r}")
    scale, size = 2, 18
    pm = QPixmap(size * scale, size * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.GlobalColor.transparent)
    color = {"recording": REC, "busy": AMBER, "loading": AMBER, "error": REC}.get(
        state, QColor("black"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(color, 1.6)
    p.setPen(pen)
    # Lips: two arcs meeting at the corners, and the line between them.
    path = QPainterPath(QPointF(2, 9))
    path.cubicTo(QPointF(5, 3.5), QPointF(13, 3.5), QPointF(16, 9))
    path.cubicTo(QPointF(13, 14.5), QPointF(5, 14.5), QPointF(2, 9))
    if state == "recording":
        p.setBrush(color)
    p.drawPath(path)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if state != "recording":
        p.drawLine(QPointF(4.5, 9), QPointF(13.5, 9))
    if state in ("paused", "error"):
        p.drawLine(QPointF(3, 16), QPointF(15, 2))
    if state in ("busy", "loading"):
        p.drawEllipse(QRectF(13.5, 1, 4, 4))
    p.end()
    icon = QIcon(pm)
    icon.setIsMask(state in ("idle", "paused"))
    return icon


def print_commit(c: Commit) -> None:
    how = "auto" if c.auto else f"picked #{c.index + 1}"
    print(f"commit [{c.mode}] {c.text!r}  ({how}; offered: {' | '.join(c.alternatives)})",
          flush=True)


def qt_defer(seconds: float, fn: Callable[[], None]) -> None:
    """Run `fn` later on the Qt main loop (never block it: the event tap lives there)."""
    QTimer.singleShot(int(seconds * 1000), fn)


class MenuBarApp(QObject):
    committed = Signal(object)      # Commit, handled by the typing stage

    def __init__(self, cfg: Config, device: str | None = None, use_lm: bool = True,
                 typing: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        a = cfg.app
        self.mode: Mode = a.mode
        self.spec = get_hotkey(a.hotkey)
        self.ptt = PushToTalk(a.min_hold_s, a.max_hold_s)
        self.capture: CaptureWorker | None = None
        self.camera_up = False
        self.face = False
        self.model_ready = False
        self.paused = False
        self.error: str | None = None
        self.pending = 0                       # clips sent to the transcriber
        self.picker: PickerModel | None = None
        self.picker_mode: Mode = self.mode
        self.last_commit: Commit | None = None
        self.hotkey_ok = False
        self.dictation_picker = cfg.app.dictation_picker
        self._warned = False

        self.transcriber = TranscribeWorker(
            cfg, device, use_lm, commands=load_command_phrases(cfg),
            hint=f"hold {self.spec.label} to talk",
        )
        self.transcriber.ready.connect(self._on_model_ready)
        self.transcriber.status.connect(self._on_worker_status)
        self.transcriber.result.connect(self._on_result)
        self.transcriber.failed.connect(self._on_failed)

        self.output: TypingOutput | None = None
        if typing and cfg.typing.enabled:
            self.output = TypingOutput(
                cfg.typing, QuartzEmitter(cfg.typing, defer=qt_defer), qt_defer,
                can_post=permissions.accessibility_granted, on_signal=self.handle_signal,
                on_warning=self._typing_warning)
            self.committed.connect(self.output)
        self.hotkey = GlobalHotkey(self.spec, self._on_key,
                                   self.output.user_input if self.output else None)
        self.popup = CursorPopup()
        self.popup.chosen.connect(lambda i: self._finish_picker(PickerResult(i)))
        self._picker_timer = QTimer(self)
        self._picker_timer.setSingleShot(True)
        self._picker_timer.timeout.connect(self._picker_timeout)
        self._hold_timer = QTimer(self)          # enforces app.max_hold_s
        self._hold_timer.timeout.connect(self._check_hold)

        self.tray = QSystemTrayIcon(tray_icon("loading"))
        self.tray.setToolTip("Mouthpiece")
        self._build_menu()

    # --- menu ------------------------------------------------------------------

    def _build_menu(self) -> None:
        m = QMenu()
        m.setFont(mono_font(12))
        self.status_action = m.addAction("starting…")
        self.status_action.setEnabled(False)
        self.weights_action = m.addAction(self._weights_text())
        self.weights_action.setEnabled(False)
        m.addSeparator()
        group = QActionGroup(m)
        group.setExclusive(True)
        self.mode_actions: dict[Mode, QAction] = {}
        for mode in MODES:
            act = m.addAction(f"{mode.capitalize()} mode")
            act.setCheckable(True)
            act.setChecked(mode == self.mode)
            act.triggered.connect(lambda _=False, md=mode: self.set_mode(md))
            group.addAction(act)
            self.mode_actions[mode] = act
        self.dictation_picker_action = m.addAction("Show top 3 in dictation")
        self.dictation_picker_action.setCheckable(True)
        self.dictation_picker_action.setChecked(self.dictation_picker)
        self.dictation_picker_action.toggled.connect(self._set_dictation_picker)
        m.addSeparator()
        self.last_action = m.addAction("last: —")
        self.last_action.setEnabled(False)
        self.pause_action = m.addAction("Pause camera")
        self.pause_action.triggered.connect(self.toggle_pause)
        m.addSeparator()
        self.perm_input = m.addAction("Grant Input Monitoring…")
        self.perm_input.triggered.connect(permissions.request_input_monitoring)
        label = "typing + picker keys" if self.output else "picker keys"
        self.perm_ax = m.addAction(f"Grant Accessibility ({label})…")
        self.perm_ax.triggered.connect(permissions.request_accessibility)
        m.aboutToShow.connect(self._refresh)
        m.addSeparator()
        quit_action = m.addAction("Quit Mouthpiece")
        quit_action.triggered.connect(self.quit)
        self.menu = m
        self.tray.setContextMenu(m)

    def _weights_text(self) -> str:
        ckpt = resolve_checkpoint(self.cfg)
        kind = "fine-tuned" if ckpt.finetuned else "pretrained"
        name = ckpt.run_dir.name if ckpt.run_dir is not None else ckpt.path.name
        return f"weights: {kind} ({name})"

    def _refresh(self) -> None:
        """Recompute icon, status line and permission items from the current state."""
        if self.error:
            state, text = "error", f"error: {self.error}"
        elif self.paused:
            state, text = "paused", "paused — camera off"
        elif not self.model_ready or not self.camera_up:
            parts = [p for p, ok in (("model", self.model_ready), ("camera", self.camera_up))
                     if not ok]
            state, text = "loading", f"starting {' + '.join(parts)}…"
        elif self.ptt.recording:
            state, text = "recording", "● listening…"
        elif self.pending:
            state, text = "busy", "reading lips…"
        elif not self.hotkey_ok:
            state, text = "error", "hotkey needs Input Monitoring permission"
        elif self.output is not None and not self.typing_ok:
            state, text = "error", "typing needs Accessibility permission"
        else:
            face = "" if self.face else " (no face in view)"
            state, text = "idle", f"ready — hold {self.spec.label} to talk{face}"
        self.tray.setIcon(tray_icon(state))
        self.status_action.setText(text)
        self.tray.setToolTip(f"Mouthpiece — {self.mode} mode — {text}")
        self.perm_input.setVisible(not self.hotkey_ok)
        self.perm_ax.setVisible(not self.typing_ok if self.output is not None
                                else self.hotkey_ok and not self.hotkey.active)
        self.pause_action.setText("Resume camera" if self.paused else "Pause camera")

    def _typing_warning(self, msg: str) -> None:
        self._warned = True
        self.popup.show_status(f"⚠ {msg}", REC, 2.5)

    def _set_dictation_picker(self, on: bool) -> None:
        self.dictation_picker = on

    @property
    def typing_ok(self) -> bool:
        return bool(permissions.accessibility_granted())

    def handle_signal(self, signal: AppSignal) -> None:
        """"stop listening" / "start listening" from the typing stage."""
        if (signal == "pause") != self.paused:
            self.toggle_pause()

    # --- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self.tray.show()
        self.transcriber.start()
        self._start_capture()
        self.hotkey_ok = self.hotkey.start()
        if not self.hotkey_ok:
            print("mouthpiece: global hotkey unavailable — grant Input Monitoring to the app "
                  "running Python (System Settings → Privacy & Security), then restart it.",
                  flush=True)
        elif not self.hotkey.active:
            print("mouthpiece: listen-only hotkey (no Accessibility): the picker accepts "
                  "clicks and the timeout, not 1/2/3 keys.", flush=True)
        if self.output is not None and not self.typing_ok:
            print("mouthpiece: typing is off until you grant Accessibility to the app running "
                  "Python (menu → Grant Accessibility…), then restart it.", flush=True)
        self._refresh()

    def _start_capture(self) -> None:
        self.camera_up = False
        self.capture = CaptureWorker(self.cfg, emit_states=False)
        self.capture.started.connect(self._on_camera_started)
        self.capture.face_changed.connect(self._on_face)
        self.capture.failed.connect(self._on_failed)
        self.capture.clip_ready.connect(self._on_clip)
        self.capture.start()

    def _stop_capture(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.camera_up = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._cancel_recording()
        if self.paused:
            self._stop_capture()
        else:
            self.error = None
            self._start_capture()
        self._refresh()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        self._refresh()

    def quit(self) -> None:
        self.hotkey.stop()
        self.popup.hide()
        self._stop_capture()
        self.transcriber.stop()
        self.tray.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # --- push-to-talk ----------------------------------------------------------

    def _on_key(self, kind: KeyEvent) -> None:
        now = time.monotonic()
        if kind is KeyEvent.DOWN:
            self.handle_ptt(self.ptt.press(now))
        elif kind is KeyEvent.UP:
            self.handle_ptt(self.ptt.release(now))
        else:
            self.handle_ptt(self.ptt.interrupt())

    def _check_hold(self) -> None:
        self.handle_ptt(self.ptt.tick(time.monotonic()))

    def handle_ptt(self, action: PttAction | None) -> None:
        if action is None:
            return
        if action is PttAction.START:
            if self.picker is not None:  # speaking again accepts what's highlighted
                self._finish_picker(PickerResult(self.picker.selected, auto=True))
            if not self.ready:
                self.ptt.reset()
                self.popup.show_status(self.status_action.text(), DIM, 1.5)
                return
            self._send("ptt_start")
            self._hold_timer.start(250)
            self.popup.show_status("● listening", REC)
        elif action is PttAction.STOP:
            self._hold_timer.stop()
            self._send("ptt_stop")
            self.pending += 1
            self.popup.show_status("… reading lips", AMBER)
        else:
            self._cancel_recording()
            self.popup.hide()
        self._refresh()

    @property
    def ready(self) -> bool:
        return (self.model_ready and self.camera_up and not self.paused
                and self.error is None and self.capture is not None)

    def _cancel_recording(self) -> None:
        self._hold_timer.stop()
        self.ptt.reset()
        self._send("ptt_cancel")

    def _send(self, cmd: str) -> None:
        if self.capture is not None:
            self.capture.commands.put(cmd)

    def _on_clip(self, frames: Any, landmarks: Any) -> None:
        self.transcriber.clips.put((frames, landmarks, self.mode))

    # --- results / picker ------------------------------------------------------

    def _on_result(self, t: Transcript, mode: str) -> None:
        self.pending = max(0, self.pending - 1)
        md: Mode = "command" if mode == "command" else "dictation"
        alts = alternatives(t, self.cfg.decode.top_k)
        self._refresh()
        if not alts:
            self.popup.show_status("∅ nothing recognised", DIM, 1.5)
            return
        switch = mode_switch(alts[0])
        if switch is not None:
            self.set_mode(switch)
            self.popup.show_status(f"→ {switch} mode", AMBER, 1.2)
            return
        self.show_picker(alts, md, t.timing.total_s)

    def show_picker(self, alts: tuple[str, ...], mode: Mode, seconds: float = 0.0) -> None:
        self.picker, self.picker_mode = PickerModel(alts), mode
        timeout = self.cfg.app.picker_timeout_s
        if not uses_picker(mode, self.dictation_picker):
            self._warned = False
            self._finish_picker(PickerResult(0, auto=True))
            if not self._warned:          # keep a typing warning on screen instead
                shown = alts[0].lower()
                shown = shown if len(shown) <= 40 else shown[:39] + "…"
                self.popup.show_status(f"✓ {shown}", DIM, 1.2)
            return
        if len(alts) == 1 and timeout > 0:
            self._finish_picker(PickerResult(0, auto=True))
            return
        keys = "1-3 · ↵ · esc" if self.hotkey.active else "click to pick"
        meta = f"{mode} · {seconds:.1f}s · {keys}"
        self.popup.show_choices(alts, meta, timeout)
        self.hotkey.key_filter = self._picker_key
        if timeout > 0:
            self._picker_timer.start(int(timeout * 1000))

    def _picker_key(self, key: str) -> bool:
        if self.picker is None:
            return False
        consumed, result = self.picker.handle_key(key)
        if result is not None:
            self._finish_picker(result)
        elif consumed:
            self.popup.set_selected(self.picker.selected, stop_countdown=self.picker.touched)
            if self.picker.touched:
                self._picker_timer.stop()
        return consumed

    def _picker_timeout(self) -> None:
        if self.picker is not None and (result := self.picker.timeout()) is not None:
            self._finish_picker(result)

    def _finish_picker(self, result: PickerResult) -> None:
        picker, self.picker = self.picker, None
        self.hotkey.key_filter = None
        self._picker_timer.stop()
        self.popup.hide()
        if picker is None:
            return
        commit = picker.commit(result, self.picker_mode)
        if commit is not None:
            self.last_commit = commit
            self.last_action.setText(f"last: {commit.text}")
            self.committed.emit(commit)

    # --- worker / camera status -----------------------------------------------

    def _on_model_ready(self, rec: Recognizer) -> None:
        self.model_ready = True
        if rec.checkpoint is not None:
            self.weights_action.setText(
                f"weights: {'fine-tuned' if rec.checkpoint.finetuned else 'pretrained'}"
                f" ({(rec.checkpoint.run_dir or rec.checkpoint.path).name}) · {rec.device.type}")
        self._refresh()

    def _on_worker_status(self, text: str) -> None:
        if text.startswith(("no usable face", "clip too short")):
            self.pending = max(0, self.pending - 1)
            self.popup.show_status(text, DIM, 2.0)
            self._refresh()

    def _on_camera_started(self) -> None:
        self.camera_up = True
        self._refresh()

    def _on_face(self, face: bool) -> None:
        self.face = face
        self._refresh()

    def _on_failed(self, msg: str) -> None:
        if self.pending and not msg.startswith("model failed"):
            self.pending -= 1            # a transcription failed; the app stays usable
            self.popup.show_status(f"error: {msg}", REC, 3.0)
        else:
            self.error = msg
        print(f"mouthpiece: {msg}", flush=True)
        self._refresh()


def hide_dock_icon() -> None:
    """Run as a menu-bar ("accessory") app: no Dock icon, no app menu."""
    try:
        import AppKit

        AppKit.NSApplication.sharedApplication().setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory)
    except ImportError:
        pass


def run_app(
    cfg: Config, device: str | None = None, use_lm: bool = True,
    on_commit: Callable[[Commit], None] | None = None, typing: bool = True,
) -> None:
    """`on_commit` gets every commit in addition to the typing stage; with typing off and no
    handler, commits are printed."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    app.setApplicationName("Mouthpiece")
    app.setQuitOnLastWindowClosed(False)
    app.setFont(mono_font(12))
    hide_dock_icon()
    if not QSystemTrayIcon.isSystemTrayAvailable():
        raise RuntimeError("no system tray / menu bar available")
    mb = MenuBarApp(cfg, device=device, use_lm=use_lm, typing=typing)
    if on_commit is not None:
        mb.committed.connect(on_commit)
    elif mb.output is None:
        mb.committed.connect(print_commit)
    mb.start()
    print(f"mouthpiece: menu-bar app running ({mb.mode} mode, hold {mb.spec.label} to talk). "
          "Quit from the menu or Ctrl-C.", flush=True)
    import signal

    signal.signal(signal.SIGINT, lambda *_: mb.quit())
    keepalive = QTimer()          # lets Python handle Ctrl-C while Qt's loop runs
    keepalive.start(250)
    keepalive.timeout.connect(lambda: None)
    app.exec()
