"""Global push-to-talk hotkey for the menu-bar app.

The pure part (key table, event classification, the hold-to-talk state machine) is testable
without macOS. `GlobalHotkey` wires it to a Quartz event tap on the main run loop, which Qt's
Cocoa event loop services. A listen-only tap needs Input Monitoring permission; an active tap
(needed to swallow the picker's 1/2/3/Return/Esc keys) needs Accessibility. Without either,
`start()` returns False and the app says which permission is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# CGEventType values (CoreGraphics/CGEventTypes.h).
KEY_DOWN = 10
KEY_UP = 11
FLAGS_CHANGED = 12
MOUSE_DOWNS = (1, 3, 25)       # left, right, other mouse down
TAP_DISABLED_BY_TIMEOUT = 0xFFFFFFFE
TAP_DISABLED_BY_USER_INPUT = 0xFFFFFFFF

# kCGEventSourceUserData of events Mouthpiece posts itself (typing.emitter.SYNTHETIC_TAG).
SYNTHETIC_TAG = 0x4D6F7574

# Modifier flags that make a key press a shortcut (cmd/ctrl/option) rather than a plain key.
FLAG_COMMAND = 0x100000
FLAG_CONTROL = 0x40000
FLAG_OPTION = 0x80000


@dataclass(frozen=True)
class HotkeySpec:
    name: str
    keycode: int          # macOS virtual key code
    flag: int = 0         # device-dependent flag bit for modifiers; 0 for ordinary keys
    label: str = ""

    @property
    def is_modifier(self) -> bool:
        return self.flag != 0


# Device-dependent bits (IOKit NX_DEVICE*KEYMASK) tell left and right modifiers apart.
HOTKEYS: dict[str, HotkeySpec] = {
    spec.name: spec
    for spec in (
        HotkeySpec("right_option", 61, 0x40, "right ⌥"),
        HotkeySpec("left_option", 58, 0x20, "left ⌥"),
        HotkeySpec("right_command", 54, 0x10, "right ⌘"),
        HotkeySpec("right_control", 62, 0x2000, "right ⌃"),
        HotkeySpec("right_shift", 60, 0x04, "right ⇧"),
        HotkeySpec("fn", 63, 0x800000, "fn"),
        HotkeySpec("f13", 105, 0, "F13"),
        HotkeySpec("f14", 107, 0, "F14"),
        HotkeySpec("f15", 113, 0, "F15"),
        HotkeySpec("f16", 106, 0, "F16"),
        HotkeySpec("f17", 64, 0, "F17"),
        HotkeySpec("f18", 79, 0, "F18"),
        HotkeySpec("f19", 80, 0, "F19"),
    )
}

# Keys the top-3 picker reacts to (main row and keypad digits, Return/Enter, Esc, arrows).
PICKER_KEYS: dict[int, str] = {
    18: "1", 19: "2", 20: "3", 83: "1", 84: "2", 85: "3",
    36: "return", 76: "return", 53: "escape", 126: "up", 125: "down",
}


def get_hotkey(name: str) -> HotkeySpec:
    try:
        return HOTKEYS[name]
    except KeyError:
        raise ValueError(f"unknown hotkey {name!r}; choose from {sorted(HOTKEYS)}") from None


class KeyEvent(Enum):
    DOWN = "down"       # the hotkey went down
    UP = "up"           # the hotkey went up
    OTHER = "other"     # some other key/modifier changed (cancels a hold: it was a shortcut)


def classify(
    spec: HotkeySpec, event_type: int, keycode: int, flags: int, autorepeat: bool = False
) -> KeyEvent | None:
    """Map one keyboard event to what it means for the hotkey (None = irrelevant)."""
    if event_type == FLAGS_CHANGED:
        if spec.is_modifier and keycode == spec.keycode:
            return KeyEvent.DOWN if flags & spec.flag else KeyEvent.UP
        return KeyEvent.OTHER
    if event_type in (KEY_DOWN, KEY_UP):
        if not spec.is_modifier and keycode == spec.keycode:
            if autorepeat:
                return None
            return KeyEvent.DOWN if event_type == KEY_DOWN else KeyEvent.UP
        return KeyEvent.OTHER if event_type == KEY_DOWN else None
    return None


class PttAction(Enum):
    START = "start"     # begin recording
    STOP = "stop"       # stop and transcribe
    CANCEL = "cancel"   # stop and discard


class PushToTalk:
    """Hold-to-talk state machine.

    Recording starts on press so no frames are lost. A release before `min_hold_s` cancels
    (a plain tap of the key, e.g. Option for something else), as does any other key while
    held (Option+E is a shortcut, not speech). `tick` stops a hold that hit `max_hold_s`.
    """

    def __init__(self, min_hold_s: float = 0.3, max_hold_s: float = 15.0) -> None:
        self.min_hold_s, self.max_hold_s = min_hold_s, max_hold_s
        self.held = False
        self._active = False   # held and still recording (not cancelled / timed out)
        self._t0 = 0.0

    @property
    def recording(self) -> bool:
        return self._active

    def press(self, t: float) -> PttAction | None:
        if self.held:
            return None
        self.held, self._active, self._t0 = True, True, t
        return PttAction.START

    def release(self, t: float) -> PttAction | None:
        if not self.held:
            return None
        self.held = False
        if not self._active:
            return None
        self._active = False
        return PttAction.CANCEL if t - self._t0 < self.min_hold_s else PttAction.STOP

    def interrupt(self) -> PttAction | None:
        if not self._active:
            return None
        self._active = False
        return PttAction.CANCEL

    def tick(self, t: float) -> PttAction | None:
        if self._active and self.max_hold_s > 0 and t - self._t0 >= self.max_hold_s:
            self._active = False
            return PttAction.STOP
        return None

    def reset(self) -> None:
        self._active = False


def picker_key(event_type: int, keycode: int, flags: int) -> str | None:
    """The picker key name for a plain key-down (no ⌘/⌃/⌥), else None."""
    if event_type != KEY_DOWN or flags & (FLAG_COMMAND | FLAG_CONTROL | FLAG_OPTION):
        return None
    return PICKER_KEYS.get(keycode)


class GlobalHotkey:
    """Quartz event tap delivering hotkey changes (and, optionally, picker keys).

    on_key(KeyEvent) is called on the main thread for DOWN/UP/OTHER. While `key_filter` is
    set, plain picker keys are passed to it; returning True swallows the key (active tap only).
    `on_input` is called for the user's own key presses and mouse clicks (not for events
    Mouthpiece posted), so the typing stage knows the cursor may have moved.
    """

    def __init__(self, spec: HotkeySpec, on_key: Callable[[KeyEvent], None],
                 on_input: Callable[[], None] | None = None) -> None:
        self.spec, self.on_key, self.on_input = spec, on_key, on_input
        self.key_filter: Callable[[str], bool] | None = None
        self.active = False            # True: can swallow keys (Accessibility granted)
        self._tap: Any = None
        self._source: Any = None
        self._callback: Any = None     # keep the PyObjC callback alive
        self._swallowed: set[int] = set()

    def start(self) -> bool:
        try:
            import Quartz as Q
        except ImportError:
            return False
        mask = (Q.CGEventMaskBit(KEY_DOWN) | Q.CGEventMaskBit(KEY_UP)
                | Q.CGEventMaskBit(FLAGS_CHANGED))
        for t in MOUSE_DOWNS:
            mask |= Q.CGEventMaskBit(t)
        self._callback = self._handle
        for option, active in ((Q.kCGEventTapOptionDefault, True),
                               (Q.kCGEventTapOptionListenOnly, False)):
            tap = Q.CGEventTapCreate(Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap, option,
                                     mask, self._callback, None)
            if tap is not None:
                self._tap, self.active = tap, active
                break
        # macOS creates a listen-only tap even without Input Monitoring, then silently
        # delivers no key events: treat that as "no hotkey" so the app can say so.
        if self._tap is not None and not self.active and not Q.CGPreflightListenEventAccess():
            self._tap = None
        if self._tap is None:
            return False
        self._source = Q.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Q.CFRunLoopAddSource(Q.CFRunLoopGetMain(), self._source, Q.kCFRunLoopCommonModes)
        Q.CGEventTapEnable(self._tap, True)
        return True

    def stop(self) -> None:
        if self._tap is None:
            return
        import Quartz as Q

        Q.CGEventTapEnable(self._tap, False)
        Q.CFRunLoopRemoveSource(Q.CFRunLoopGetMain(), self._source, Q.kCFRunLoopCommonModes)
        self._tap = self._source = None

    def _handle(self, proxy: Any, event_type: int, event: Any, refcon: Any) -> Any:
        import Quartz as Q

        if event_type in (TAP_DISABLED_BY_TIMEOUT, TAP_DISABLED_BY_USER_INPUT):
            if self._tap is not None:
                Q.CGEventTapEnable(self._tap, True)
            return event
        try:
            if Q.CGEventGetIntegerValueField(event, Q.kCGEventSourceUserData) == SYNTHETIC_TAG:
                return event          # our own typing: not the user, not the hotkey
            if event_type in MOUSE_DOWNS:
                if self.on_input is not None:
                    self.on_input()
                return event
            keycode = int(Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode))
            flags = int(Q.CGEventGetFlags(event))
            repeat = bool(Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventAutorepeat))
            if self.active and event_type == KEY_UP and keycode in self._swallowed:
                self._swallowed.discard(keycode)
                return None
            filt = self.key_filter
            name = picker_key(event_type, keycode, flags)
            # Listen-only taps can't swallow keys, so the picker then ignores the keyboard
            # (a "1" would also be typed into the focused app).
            if self.active and filt is not None and name is not None and filt(name):
                self._swallowed.add(keycode)
                return None
            kind = classify(self.spec, event_type, keycode, flags, repeat)
            if kind is not None:
                self.on_key(kind)
            if kind is KeyEvent.OTHER and event_type == KEY_DOWN and self.on_input is not None:
                self.on_input()
        except Exception as exc:  # never let an exception escape into the event tap
            print(f"mouthpiece: hotkey handler error: {exc}")
        return event
