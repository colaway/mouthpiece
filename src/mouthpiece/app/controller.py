"""UI-independent logic of the menu-bar app: modes, command phrases, the picker, commits.

`Commit` is the hand-off to the output stage: the menu-bar app emits one per accepted
utterance (`MenuBarApp.committed`), and Phase 6 turns it into keystrokes or an action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mouthpiece.config import Config
from mouthpiece.decode.recognizer import Transcript
from mouthpiece.metrics import normalize_text
from mouthpiece.record.prompts import COMMAND_PHRASES, load_prompt_file

Mode = Literal["command", "dictation"]
MODES: tuple[Mode, ...] = ("command", "dictation")

# Spoken mode switches, handled by the app itself rather than passed on as a commit.
MODE_SWITCHES: dict[str, Mode] = {"command mode": "command", "dictation mode": "dictation"}


@dataclass(frozen=True)
class Commit:
    """An accepted utterance, ready for the output stage (Phase 6)."""

    text: str
    mode: Mode
    alternatives: tuple[str, ...]   # what the picker offered, best-first
    index: int                      # which alternative was chosen
    auto: bool                      # accepted by timeout / single option, not by the user


def load_command_phrases(cfg: Config) -> list[str]:
    """Built-in command phrases plus `app.commands` (deduped by normalized text)."""
    phrases = list(COMMAND_PHRASES)
    if cfg.app.commands:
        phrases += [p.text for p in load_prompt_file(cfg.resolve(cfg.app.commands))]
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        key = normalize_text(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def uses_picker(mode: Mode, dictation_picker: bool) -> bool:
    """Command mode always shows the top 3; dictation only if `app.dictation_picker`
    (otherwise #1 is typed as soon as it is decoded; fix mistakes with "scratch that")."""
    return mode == "command" or dictation_picker


def mode_switch(text: str) -> Mode | None:
    key = normalize_text(text)
    return next((m for phrase, m in MODE_SWITCHES.items() if normalize_text(phrase) == key), None)


def alternatives(t: Transcript, limit: int = 3) -> tuple[str, ...]:
    """Distinct non-empty hypotheses, best-first."""
    seen: set[str] = set()
    out: list[str] = []
    for n in t.nbest:
        key = normalize_text(n.text)
        if key and key not in seen:
            seen.add(key)
            out.append(n.text.strip())
        if len(out) == limit:
            break
    return tuple(out)


@dataclass(frozen=True)
class PickerResult:
    index: int | None       # None: cancelled
    auto: bool = False


class PickerModel:
    """Top-3 picker state: number keys pick, arrows move, Return accepts, Esc cancels."""

    def __init__(self, options: Sequence[str]) -> None:
        if not options:
            raise ValueError("picker needs at least one option")
        self.options = tuple(options)
        self.selected = 0
        self.touched = False    # the user interacted, so stop the auto-accept countdown

    def handle_key(self, key: str) -> tuple[bool, PickerResult | None]:
        """(consumed, result). Unknown keys and out-of-range digits are not consumed."""
        n = len(self.options)
        if key.isdigit():
            i = int(key) - 1
            if 0 <= i < n:
                return True, PickerResult(i)
            return False, None
        if key in ("up", "down"):
            self.selected = (self.selected + (1 if key == "down" else -1)) % n
            self.touched = True
            return True, None
        if key == "return":
            return True, PickerResult(self.selected)
        if key == "escape":
            return True, PickerResult(None)
        return False, None

    def timeout(self) -> PickerResult | None:
        """Auto-accept the highlighted option unless the user is choosing."""
        return None if self.touched else PickerResult(self.selected, auto=True)

    def commit(self, result: PickerResult, mode: Mode) -> Commit | None:
        if result.index is None:
            return None
        return Commit(self.options[result.index], mode, self.options, result.index, result.auto)
