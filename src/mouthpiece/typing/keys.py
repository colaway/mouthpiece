"""Key combos and media keys, as data (no macOS imports).

Key codes are macOS virtual key codes for the ANSI (US) layout. Shortcuts are posted by key
code, so on a non-US layout a combo like ⌘Z follows the physical key position.
"""

from __future__ import annotations

from dataclasses import dataclass

KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45,
    "m": 46, ".": 47, "`": 50,
    "return": 36, "tab": 48, "space": 49, "backspace": 51, "escape": 53,
    "f11": 103, "home": 115, "pageup": 116, "forwarddelete": 117, "end": 119,
    "pagedown": 121, "left": 123, "right": 124, "down": 125, "up": 126,
}

# CGEventFlags masks.
MODIFIER_FLAGS: dict[str, int] = {
    "shift": 0x20000, "ctrl": 0x40000, "opt": 0x80000, "cmd": 0x100000, "fn": 0x800000,
}
_MOD_ALIASES = {"command": "cmd", "control": "ctrl", "option": "opt", "alt": "opt"}
_MOD_ORDER = ("ctrl", "opt", "shift", "cmd", "fn")
_MOD_SYMBOLS = {"ctrl": "⌃", "opt": "⌥", "shift": "⇧", "cmd": "⌘", "fn": "fn "}

# Arrow/navigation/function keys carry these flags when typed on a real keyboard; some apps
# check them, so synthetic presses do too.
_FN_KEYS = {"f11", "home", "pageup", "forwarddelete", "end", "pagedown",
            "left", "right", "down", "up"}
_NUMPAD_KEYS = {"left", "right", "down", "up"}
FLAG_NUMPAD = 0x200000

# NX_KEYTYPE_* (IOKit/hidsystem/ev_keymap.h): posted as NSSystemDefined events.
MEDIA_KEYS: dict[str, int] = {"volume_up": 0, "volume_down": 1, "mute": 7, "play_pause": 16}


@dataclass(frozen=True)
class KeyCombo:
    key: str                         # a KEYCODES name
    mods: tuple[str, ...] = ()       # canonical modifier names, in _MOD_ORDER

    @property
    def keycode(self) -> int:
        return KEYCODES[self.key]

    @property
    def flags(self) -> int:
        f = 0
        for m in self.mods:
            f |= MODIFIER_FLAGS[m]
        if self.key in _FN_KEYS:
            f |= MODIFIER_FLAGS["fn"]
        if self.key in _NUMPAD_KEYS:
            f |= FLAG_NUMPAD
        return f

    def __str__(self) -> str:
        return "".join(_MOD_SYMBOLS[m] for m in self.mods) + self.key


def combo(spec: str) -> KeyCombo:
    """Parse "cmd+shift+z" (modifiers in any order; key last)."""
    parts = [p.strip().lower() for p in spec.split("+")]
    if not parts or not parts[-1]:
        raise ValueError(f"empty key in combo {spec!r}")
    *mods, key = parts
    if key == "plus":
        key = "="
    if key not in KEYCODES:
        raise ValueError(f"unknown key {key!r} in combo {spec!r}")
    canon: set[str] = set()
    for m in mods:
        m = _MOD_ALIASES.get(m, m)
        if m not in MODIFIER_FLAGS:
            raise ValueError(f"unknown modifier {m!r} in combo {spec!r}")
        canon.add(m)
    return KeyCombo(key, tuple(m for m in _MOD_ORDER if m in canon))


BACKSPACE = combo("backspace")
RETURN = combo("return")
