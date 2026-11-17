"""What each command phrase does.

Two layers of plain data:
- *Steps* are what the emitter executes: type text, press a combo, a media key, scroll,
  launch an app, wait.
- *Actions* are what a command phrase means to the composer, which turns them into steps
  while tracking spacing, casing and what "scratch that" can remove.

`COMMAND_ACTIONS` covers every built-in phrase in `record.prompts.COMMAND_PHRASES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mouthpiece.metrics import normalize_text
from mouthpiece.typing.keys import MEDIA_KEYS, KeyCombo, combo

# --- steps (emitter level) ---------------------------------------------------


@dataclass(frozen=True)
class TypeText:
    text: str


@dataclass(frozen=True)
class Press:
    combo: KeyCombo
    repeat: int = 1


@dataclass(frozen=True)
class Media:
    key: str                # a keys.MEDIA_KEYS name

    def __post_init__(self) -> None:
        if self.key not in MEDIA_KEYS:
            raise ValueError(f"unknown media key {self.key!r}")


@dataclass(frozen=True)
class Scroll:
    lines: int              # > 0 scrolls up


@dataclass(frozen=True)
class Launch:
    target: str             # "app:<Name>" or "default:<url scheme>" (the scheme's default app)


@dataclass(frozen=True)
class Wait:
    seconds: float


Step = TypeText | Press | Media | Scroll | Launch | Wait


# --- actions (composer level) ------------------------------------------------


@dataclass(frozen=True)
class Keys:
    """Press combos in order. `keeps_history`: the text Mouthpiece typed is untouched and the
    cursor did not move (copy, bold, save), so "scratch that" can still remove it."""

    combos: tuple[KeyCombo, ...]
    keeps_history: bool = False


@dataclass(frozen=True)
class Insert:
    """Punctuation typed into the text flow.

    `space_before`: separate it from the previous word (open quote) or attach (period).
    `space_after`: the next word gets a leading space (period) or attaches (open quote).
    """

    text: str
    space_before: bool = False
    space_after: bool = True
    ends_sentence: bool = False


@dataclass(frozen=True)
class Break:
    """Return pressed `count` times; the next word starts a sentence."""

    count: int = 1


CasingStyle = Literal["capital", "upper", "lower"]


@dataclass(frozen=True)
class Casing:
    """Applies to the next dictated utterance: capital first letter, ALL CAPS, or no
    automatic capital."""

    style: CasingStyle


@dataclass(frozen=True)
class Scratch:
    """Remove the last thing Mouthpiece typed. `fallback`: with nothing to remove, press
    Backspace once (deletes a selection)."""

    fallback: bool = False


@dataclass(frozen=True)
class Say:
    """Type a fixed phrase as if it were dictated."""

    text: str


@dataclass(frozen=True)
class System:
    """App launches, media keys, scrolling: no effect on the text state."""

    steps: tuple[Step, ...]


AppSignal = Literal["pause", "resume"]


@dataclass(frozen=True)
class Internal:
    """Handled by the app itself (pause/resume the camera)."""

    signal: AppSignal


@dataclass(frozen=True)
class Ignore:
    reason: str


Action = Keys | Insert | Break | Casing | Scratch | Say | System | Internal | Ignore


def keys(*specs: str, keeps_history: bool = False) -> Keys:
    return Keys(tuple(combo(s) for s in specs), keeps_history)


def _press(*specs: str) -> System:
    return System(tuple(Press(combo(s)) for s in specs))


_ACTIONS: dict[str, Action] = {
    # editing
    "new line": Break(1),
    "new paragraph": Break(2),
    "select all": keys("cmd+a"),
    "copy that": keys("cmd+c", keeps_history=True),
    "paste it": keys("cmd+v"),
    "cut that": keys("cmd+x"),
    "undo that": keys("cmd+z"),
    "redo that": keys("cmd+shift+z"),
    "delete that": Scratch(fallback=True),
    "scratch that": Scratch(),
    "delete the last word": keys("opt+backspace"),
    "bold that": keys("cmd+b", keeps_history=True),
    "italic that": keys("cmd+i", keeps_history=True),
    "save the file": keys("cmd+s", keeps_history=True),
    # navigation / selection
    "go to the end": keys("cmd+down"),
    "go to the start": keys("cmd+up"),
    "go back": keys("cmd+["),
    "go forward": keys("cmd+]"),
    "page up": keys("pageup"),
    "page down": keys("pagedown"),
    "move left": keys("left"),
    "move right": keys("right"),
    "move up": keys("up"),
    "move down": keys("down"),
    "select the line": keys("cmd+left", "cmd+shift+right"),
    "select the word": keys("opt+left", "opt+shift+right"),
    "select the next word": keys("opt+shift+right"),
    "find on page": keys("cmd+f"),
    # tabs / windows / apps
    "next tab": keys("ctrl+tab"),
    "previous tab": keys("ctrl+shift+tab"),
    "close tab": keys("cmd+w"),
    "open a new tab": keys("cmd+t"),
    "close window": keys("cmd+shift+w"),
    "switch app": keys("cmd+tab"),
    # keys
    "send it": keys("return"),
    "press enter": keys("return"),
    "cancel": keys("escape"),
    "press escape": keys("escape"),
    "press tab": keys("tab"),
    "press space": Insert(" ", space_after=False),
    # casing
    "capital letter": Casing("capital"),
    "all caps": Casing("upper"),
    "no caps": Casing("lower"),
    # punctuation
    "period": Insert(".", ends_sentence=True),
    "comma": Insert(","),
    "question mark": Insert("?", ends_sentence=True),
    "exclamation mark": Insert("!", ends_sentence=True),
    "open quote": Insert('"', space_before=True, space_after=False),
    "close quote": Insert('"'),
    "open bracket": Insert("(", space_before=True, space_after=False),
    "close bracket": Insert(")"),
    # system
    "scroll up": System((Scroll(5),)),
    "scroll down": System((Scroll(-5),)),
    "search the web": System((Launch("default:https"), Wait(0.8), Press(combo("cmd+t")))),
    "open the browser": System((Launch("default:https"),)),
    "open the terminal": System((Launch("app:Terminal"),)),
    "open my mail": System((Launch("default:mailto"),)),
    "show the desktop": _press("f11"),
    "volume up": System((Media("volume_up"),)),
    "volume down": System((Media("volume_down"),)),
    "mute": System((Media("mute"),)),
    "play music": System((Media("play_pause"),)),
    "pause music": System((Media("play_pause"),)),
    "take a screenshot": _press("cmd+shift+4"),
    "zoom in": keys("cmd+=", keeps_history=True),
    "zoom out": keys("cmd+-", keeps_history=True),
    # the app
    "stop listening": Internal("pause"),
    "start listening": Internal("resume"),
    "dictation mode": Ignore("mode switch (handled by the app)"),
    "command mode": Ignore("mode switch (handled by the app)"),
    # short replies, typed as text
    "yes": Say("yes"),
    "no": Say("no"),
    "okay": Say("okay"),
    "thank you": Say("thank you"),
    "sounds good": Say("sounds good"),
}

COMMAND_ACTIONS: dict[str, Action] = {normalize_text(k): v for k, v in _ACTIONS.items()}

# Editing phrases that act as commands in dictation mode too, when they are the whole
# utterance ("scratch that" fixes a mistake without switching modes). Inside a longer
# sentence they stay ordinary words ("a trial period").
DICTATION_COMMANDS: frozenset[str] = frozenset(normalize_text(p) for p in (
    "scratch that", "delete that", "new line", "new paragraph",
    "period", "comma", "question mark", "exclamation mark",
))


def action_for(phrase: str) -> Action:
    """The action for a command phrase; phrases without one (e.g. user-added commands) are
    ignored."""
    return COMMAND_ACTIONS.get(normalize_text(phrase),
                               Ignore("no action for this phrase"))
