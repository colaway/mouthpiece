"""Turn commits into steps: dictation formatting, command actions, "scratch that".

Pure (no macOS). The composer only knows what Mouthpiece itself typed: it tracks whether the
next word starts a sentence and needs a leading space, a pending casing modifier, and a
history of typed chunks so "scratch that" can backspace over the last one. Anything that may
move the cursor (navigation keys, the user typing or clicking) forgets that history, so
"scratch that" never deletes text Mouthpiece did not type.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace

from mouthpiece.metrics import normalize_text
from mouthpiece.typing.actions import (
    DICTATION_COMMANDS,
    Action,
    AppSignal,
    Break,
    Casing,
    CasingStyle,
    Ignore,
    Insert,
    Internal,
    Keys,
    Press,
    Say,
    Scratch,
    Step,
    System,
    TypeText,
    action_for,
)
from mouthpiece.typing.keys import BACKSPACE, RETURN

_PRONOUN_I = re.compile(r"\bi\b")     # i, i'm, i'll, i've, i'd (the apostrophe is a boundary)


@dataclass(frozen=True)
class TypingState:
    capitalize_next: bool = True        # the next word starts a sentence
    space_next: bool = False            # the next word needs a leading space
    casing: CasingStyle | None = None   # modifier for the next utterance


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...] = ()
    signal: AppSignal | None = None        # for the app (pause / resume)
    note: str = ""                      # one line for the log


def format_words(raw: str, capitalize: bool, casing: CasingStyle | None = None) -> str:
    """LRS3 UPPERCASE text → lower case with "I" fixed, then the casing rules.

    `capitalize`: sentence start. "capital" forces a capital first letter, "lower" suppresses
    the automatic one, "upper" is ALL CAPS.
    """
    text = " ".join(raw.replace("’", "'").split()).lower()
    if not text:
        return ""
    if casing == "upper":
        return text.upper()
    text = _PRONOUN_I.sub("I", text)
    if casing == "capital" or (capitalize and casing != "lower"):
        text = text[0].upper() + text[1:]
    return text


class Composer:
    def __init__(self, history: int = 20) -> None:
        self.state = TypingState()
        # (characters typed, state before): what "scratch that" undoes, most recent last.
        self.history: deque[tuple[int, TypingState]] = deque(maxlen=max(1, history))
        self._saved: tuple[TypingState, list[tuple[int, TypingState]]] | None = None

    # --- public ----------------------------------------------------------------

    def plan(self, text: str, mode: str) -> Plan:
        """Steps for one commit (`mode` "dictation" or "command"); updates the state.
        `rollback()` undoes the last call."""
        self._saved = (self.state, list(self.history))
        if mode == "dictation" and normalize_text(text) not in DICTATION_COMMANDS:
            return self._dictate(text)
        return self._apply(action_for(text), text)

    def rollback(self) -> None:
        """Forget the last `plan()` (its steps were not executed)."""
        if self._saved is not None:
            self.state, hist = self._saved
            self.history.clear()
            self.history.extend(hist)
            self._saved = None

    def forget(self) -> None:
        """The cursor may have moved (user typed/clicked): nothing is safe to scratch."""
        self.history.clear()

    def reset(self) -> None:
        self.state = TypingState()
        self.history.clear()

    # --- internals -------------------------------------------------------------

    def _typed(self, text: str, after: TypingState) -> TypeText:
        self.history.append((len(text), self.state))
        self.state = after
        return TypeText(text)

    def _dictate(self, raw: str) -> Plan:
        s = self.state
        words = format_words(raw, s.capitalize_next, s.casing)
        if not words:
            return Plan(note="nothing to type")
        text = (" " if s.space_next else "") + words
        return Plan((self._typed(text, TypingState(False, True, None)),), note=repr(text))

    def _apply(self, action: Action, phrase: str) -> Plan:
        s = self.state
        if isinstance(action, Say):
            return self._dictate(action.text)
        if isinstance(action, Insert):
            text = (" " if s.space_next and action.space_before else "") + action.text
            after = TypingState(
                capitalize_next=action.ends_sentence or s.capitalize_next,
                space_next=action.space_after, casing=s.casing)
            return Plan((self._typed(text, after),), note=f"insert {text!r}")
        if isinstance(action, Break):
            self.history.append((action.count, s))
            self.state = TypingState(True, False, s.casing)
            return Plan((Press(RETURN, action.count),), note=f"return ×{action.count}")
        if isinstance(action, Scratch):
            if self.history:
                n, before = self.history.pop()
                self.state = before
                return Plan((Press(BACKSPACE, n),), note=f"scratch {n} chars")
            if action.fallback:
                return Plan((Press(BACKSPACE),), note="backspace")
            return Plan(note="nothing to scratch")
        if isinstance(action, Keys):
            if not action.keeps_history:
                self.history.clear()
            return Plan(tuple(Press(c) for c in action.combos),
                        note=" ".join(str(c) for c in action.combos))
        if isinstance(action, Casing):
            self.state = replace(s, casing=action.style)
            return Plan(note=f"next utterance: {action.style}")
        if isinstance(action, System):
            return Plan(action.steps, note=phrase)
        if isinstance(action, Internal):
            return Plan(signal=action.signal, note=action.signal)
        assert isinstance(action, Ignore)
        return Plan(note=f"ignored ({action.reason})")
