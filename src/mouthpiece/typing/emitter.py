"""Executing steps: the `Emitter` interface, a recording fake, and the macOS implementation.

`QuartzEmitter` posts CGEvents: text as unicode keyboard events (no key codes, so any
character and any keyboard layout works), with a clipboard + ⌘V fallback for long text and
for apps listed in `typing.paste_apps`; the old clipboard is put back afterwards. Posting
events needs Accessibility permission for the app running Python; without it macOS drops
them silently, so callers check `permissions.accessibility_granted()` first.

Every posted event carries `SYNTHETIC_TAG` in its user-data field, so Mouthpiece's own event
tap can tell them from the user's keystrokes.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from mouthpiece.app.hotkey import SYNTHETIC_TAG
from mouthpiece.config import TypingConfig
from mouthpiece.typing.actions import Launch, Media, Press, Scroll, Step, TypeText, Wait
from mouthpiece.typing.keys import MEDIA_KEYS, KeyCombo, combo

UNICODE_CHUNK = 20                  # UTF-16 units per event (CGEventKeyboardSetUnicodeString)
TRANSIENT_TYPE = "org.nspasteboard.TransientType"   # clipboard managers: don't record this

Defer = Callable[[float, Callable[[], None]], None]


def blocking_defer(seconds: float, fn: Callable[[], None]) -> None:
    """For scripts without an event loop. Never use it on the menu-bar app's main thread:
    its event tap would hold our own posted events until the sleep ends."""
    time.sleep(seconds)
    fn()


class Emitter(Protocol):
    def type_text(self, text: str) -> None: ...
    def press(self, combo: KeyCombo) -> None: ...
    def media(self, key: str) -> None: ...
    def scroll(self, lines: int) -> None: ...
    def launch(self, target: str) -> None: ...


def run_steps(steps: Sequence[Step], emitter: Emitter, defer: Defer) -> None:
    """Execute steps in order; a `Wait` schedules the rest through `defer`."""
    for i, step in enumerate(steps):
        if isinstance(step, Wait):
            _defer_rest(steps[i + 1:], emitter, defer, step.seconds)
            return
        if isinstance(step, TypeText):
            emitter.type_text(step.text)
        elif isinstance(step, Press):
            for _ in range(step.repeat):
                emitter.press(step.combo)
        elif isinstance(step, Media):
            emitter.media(step.key)
        elif isinstance(step, Scroll):
            emitter.scroll(step.lines)
        else:
            assert isinstance(step, Launch)
            emitter.launch(step.target)


def _defer_rest(rest: Sequence[Step], emitter: Emitter, defer: Defer, seconds: float) -> None:
    def go() -> None:
        run_steps(rest, emitter, defer)

    defer(seconds, go)


def utf16_chunks(text: str, size: int = UNICODE_CHUNK) -> list[str]:
    """Split into pieces of at most `size` UTF-16 code units, never inside a character."""
    chunks: list[str] = []
    cur, units = "", 0
    for ch in text:
        n = 2 if ord(ch) > 0xFFFF else 1
        if units + n > size and cur:
            chunks.append(cur)
            cur, units = "", 0
        cur += ch
        units += n
    if cur:
        chunks.append(cur)
    return chunks


def use_paste(text: str, cfg: TypingConfig, frontmost: str | None) -> bool:
    if cfg.method == "paste":
        return True
    if cfg.method == "keys":
        return False
    return len(text) > cfg.paste_over_chars or (frontmost is not None
                                                 and frontmost in cfg.paste_apps)


class RecordingEmitter:
    """Records calls and simulates a text field (`buffer`), for tests and `--dry-run`."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.buffer = ""

    def type_text(self, text: str) -> None:
        self.calls.append(f"type {text!r}")
        self.buffer += text

    def press(self, combo: KeyCombo) -> None:
        self.calls.append(f"press {combo}")
        if not combo.mods and combo.key == "backspace":
            self.buffer = self.buffer[:-1]
        elif not combo.mods and combo.key == "return":
            self.buffer += "\n"

    def media(self, key: str) -> None:
        self.calls.append(f"media {key}")

    def scroll(self, lines: int) -> None:
        self.calls.append(f"scroll {lines:+d}")

    def launch(self, target: str) -> None:
        self.calls.append(f"launch {target}")


# --- macOS --------------------------------------------------------------------


class Pasteboard:
    """Save, replace and restore the general pasteboard (or one passed in, for tests)."""

    def __init__(self, pb: Any = None) -> None:
        import AppKit

        self.pb = pb if pb is not None else AppKit.NSPasteboard.generalPasteboard()

    def snapshot(self) -> list[dict[str, Any]]:
        items = self.pb.pasteboardItems() or []
        out: list[dict[str, Any]] = []
        for item in items:
            data = {str(t): item.dataForType_(t) for t in item.types()}
            out.append({t: d for t, d in data.items() if d is not None})
        return out

    def set_text(self, text: str) -> int:
        """Replace the contents with `text` (marked transient); returns the change count."""
        import AppKit

        self.pb.clearContents()
        self.pb.declareTypes_owner_([AppKit.NSPasteboardTypeString, TRANSIENT_TYPE], None)
        self.pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
        self.pb.setString_forType_("", TRANSIENT_TYPE)
        return int(self.pb.changeCount())

    def restore(self, saved: list[dict[str, Any]], expected_count: int) -> bool:
        """Put `saved` back unless something else changed the pasteboard meanwhile."""
        import AppKit

        if int(self.pb.changeCount()) != expected_count:
            return False
        self.pb.clearContents()
        items = []
        for entry in saved:
            item = AppKit.NSPasteboardItem.alloc().init()
            for t, d in entry.items():
                item.setData_forType_(d, t)
            items.append(item)
        if items:
            self.pb.writeObjects_(items)
        return True


def frontmost_bundle_id() -> str | None:
    try:
        import AppKit

        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return str(app.bundleIdentifier()) if app is not None else None
    except Exception:
        return None


class QuartzEmitter:
    def __init__(self, cfg: TypingConfig, defer: Defer = blocking_defer,
                 pasteboard: Pasteboard | None = None) -> None:
        import Quartz as Q

        self.cfg, self.defer = cfg, defer
        self.Q = Q
        # A private source: our events don't inherit modifiers the user is holding (e.g. the
        # push-to-talk ⌥ while the picker auto-accepts).
        self.source = Q.CGEventSourceCreate(Q.kCGEventSourceStatePrivate)
        self._pasteboard = pasteboard
        self._paste_v = combo("cmd+v")

    # building (separate from posting so tests can inspect events)

    def key_event(self, keycode: int, down: bool, flags: int = 0) -> Any:
        Q = self.Q
        ev = Q.CGEventCreateKeyboardEvent(self.source, keycode, down)
        Q.CGEventSetFlags(ev, flags)
        Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, SYNTHETIC_TAG)
        return ev

    def unicode_events(self, text: str) -> list[Any]:
        out = []
        for chunk in utf16_chunks(text):
            for down in (True, False):
                ev = self.key_event(0, down)
                self.Q.CGEventKeyboardSetUnicodeString(ev, len(chunk.encode("utf-16-le")) // 2,
                                                       chunk)
                out.append(ev)
        return out

    def _post(self, ev: Any) -> None:
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)
        if self.cfg.key_delay_ms > 0:
            time.sleep(self.cfg.key_delay_ms / 1000)

    # Emitter

    def type_text(self, text: str) -> None:
        if not text:
            return
        if use_paste(text, self.cfg, frontmost_bundle_id()):
            self.paste(text)
            return
        for ev in self.unicode_events(text):
            self._post(ev)

    def paste(self, text: str) -> None:
        pb = self._pasteboard or Pasteboard()
        saved = pb.snapshot()
        count = pb.set_text(text)
        self.press(self._paste_v)

        def restore() -> None:
            pb.restore(saved, count)

        self.defer(self.cfg.clipboard_restore_s, restore)

    def press(self, combo: KeyCombo) -> None:
        for down in (True, False):
            self._post(self.key_event(combo.keycode, down, combo.flags))

    def media(self, key: str) -> None:
        import AppKit

        code = MEDIA_KEYS[key]
        for state in (0xA, 0xB):            # key down, key up
            ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(  # noqa: E501
                AppKit.NSEventTypeSystemDefined, (0, 0), state << 8, 0, 0, None, 8,
                (code << 16) | (state << 8), -1)
            cg = ev.CGEvent()
            self.Q.CGEventSetIntegerValueField(cg, self.Q.kCGEventSourceUserData, SYNTHETIC_TAG)
            self._post(cg)

    def scroll(self, lines: int) -> None:
        Q = self.Q
        ev = Q.CGEventCreateScrollWheelEvent(self.source, Q.kCGScrollEventUnitLine, 1, lines)
        Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, SYNTHETIC_TAG)
        self._post(ev)

    def launch(self, target: str) -> None:
        cmd = launch_command(target)
        if cmd is None:
            print(f"mouthpiece: can't launch {target!r}", flush=True)
            return
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def default_app_path(scheme: str) -> str | None:
    """The app that opens `scheme:` URLs (https → default browser, mailto → mail app)."""
    try:
        import AppKit

        url = AppKit.NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(
            AppKit.NSURL.URLWithString_(f"{scheme}:"))
        return str(url.path()) if url is not None else None
    except Exception:
        return None


def launch_command(target: str, resolve: Callable[[str], str | None] = default_app_path,
                   ) -> list[str] | None:
    kind, _, name = target.partition(":")
    if kind == "app" and name:
        return ["open", "-a", name]
    if kind == "default" and name:
        path = resolve(name)
        return ["open", path] if path else None
    return None
