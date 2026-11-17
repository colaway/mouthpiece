"""Phrases to mouth while recording, and the order they are prompted in.

Two built-in sets match the app's two modes: short editing/navigation commands and
free-form dictation sentences (written for this project). A user file adds more lines.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mouthpiece.metrics import normalize_text

PromptKind = Literal["command", "dictation", "custom"]

COMMAND_PHRASES: tuple[str, ...] = (
    "new line", "new paragraph", "select all", "copy that", "paste it", "cut that",
    "undo that", "redo that", "delete that", "delete the last word", "scratch that",
    "go to the end", "go to the start", "go back", "go forward", "scroll up", "scroll down",
    "page up", "page down", "next tab", "previous tab", "close tab", "open a new tab",
    "close window", "save the file", "send it", "cancel", "stop listening", "start listening",
    "press enter", "press escape", "press tab", "press space", "capital letter",
    "all caps", "no caps", "period", "comma", "question mark", "exclamation mark",
    "open quote", "close quote", "open bracket", "close bracket", "bold that",
    "italic that", "move left", "move right", "move up", "move down", "select the line",
    "select the word", "select the next word", "find on page", "search the web",
    "open the browser", "open the terminal", "open my mail", "switch app", "show the desktop",
    "volume up", "volume down", "mute", "play music", "pause music", "take a screenshot",
    "zoom in", "zoom out", "dictation mode", "command mode", "yes", "no", "okay",
    "thank you", "sounds good",
)

DICTATION_SENTENCES: tuple[str, ...] = (
    "I will be there in about ten minutes",
    "can we move the meeting to tomorrow morning",
    "thanks for sending this over so quickly",
    "let me know if you have any questions",
    "I am running a little late today",
    "the build is green and ready for review",
    "please take a look when you have a moment",
    "I think we should try a different approach",
    "could you share the notes from the call",
    "I just pushed a fix for that bug",
    "what time works best for you this week",
    "sorry I missed your message earlier",
    "the train was delayed again this morning",
    "we need more coffee in the kitchen",
    "I will send the report by the end of the day",
    "that sounds like a great idea to me",
    "remind me to call the doctor on monday",
    "the weather is lovely outside right now",
    "I am working from home tomorrow",
    "happy birthday I hope you have a wonderful day",
    "the package should arrive on thursday",
    "do you want to grab lunch later",
    "I have read the document and left some comments",
    "the tests pass on my machine",
    "let us keep this short and simple",
    "my battery is almost empty",
    "I am not sure that is the right answer",
    "we can talk about it after the break",
    "please turn off the lights when you leave",
    "the store closes at nine tonight",
    "I need to finish this before the deadline",
    "he said the numbers look much better now",
    "she will join the call in five minutes",
    "can you help me move the couch on saturday",
    "I left my keys on the kitchen table",
    "the new version is much faster than the old one",
    "write the first draft and we will edit it together",
    "the cat is sleeping on my keyboard again",
    "I would like a glass of water please",
    "we should plan a trip for the summer",
    "turn left at the next traffic light",
    "the movie starts at half past seven",
    "I forgot to buy milk on the way home",
    "this morning I went for a long walk",
    "the meeting notes are in the shared folder",
    "I agree with most of what you said",
    "it might rain later so bring an umbrella",
    "how was your weekend",
    "the server needs to be restarted tonight",
    "I will review the pull request after lunch",
    "everything is working as expected",
    "please do not forget to water the plants",
    "the kids are playing in the garden",
    "my phone is on silent so send me a message",
    "we are almost out of time for today",
    "I really enjoyed the book you lent me",
    "the dinner was delicious thank you",
    "let me check my calendar and get back to you",
    "the screen is too bright at night",
    "good morning everyone and welcome back",
)


@dataclass(frozen=True)
class Prompt:
    text: str
    kind: PromptKind

    @property
    def key(self) -> str:
        """Normalized text: what counts as "the same phrase" across takes."""
        return normalize_text(self.text)


def builtin_prompts(kinds: Iterable[PromptKind] = ("command", "dictation")) -> list[Prompt]:
    sets: dict[PromptKind, tuple[str, ...]] = {
        "command": COMMAND_PHRASES, "dictation": DICTATION_SENTENCES,
    }
    return [Prompt(t, k) for k in kinds for t in sets.get(k, ())]


def load_prompt_file(path: str | Path) -> list[Prompt]:
    """One phrase per line; blank lines and `#` comments ignored."""
    out: list[Prompt] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and normalize_text(text):
            out.append(Prompt(text, "custom"))
    return out


def dedupe(prompts: Iterable[Prompt]) -> list[Prompt]:
    seen: set[str] = set()
    out: list[Prompt] = []
    for p in prompts:
        if p.key and p.key not in seen:
            seen.add(p.key)
            out.append(p)
    return out


class PromptScheduler:
    """Hands out the phrase with the fewest takes so far (random among ties).

    Phrases that reached `takes_per_phrase` (counting earlier sessions) or were skipped this
    session are no longer offered; `next()` returns None once nothing is left.
    """

    def __init__(
        self,
        prompts: Iterable[Prompt],
        existing: Mapping[str, int] | None = None,
        takes_per_phrase: int = 5,
        seed: int | None = None,
    ) -> None:
        self.prompts = dedupe(prompts)
        self.counts: Counter[str] = Counter(existing or {})
        self.takes_per_phrase = takes_per_phrase
        self.skipped: set[str] = set()
        self._rng = random.Random(seed)
        self._last: str | None = None

    def remaining(self) -> list[Prompt]:
        return [p for p in self.prompts
                if p.key not in self.skipped and self.counts[p.key] < self.takes_per_phrase]

    def next(self) -> Prompt | None:
        left = self.remaining()
        if not left:
            return None
        if len(left) > 1:  # avoid the same phrase twice in a row
            left = [p for p in left if p.key != self._last] or left
        low = min(self.counts[p.key] for p in left)
        choice = self._rng.choice([p for p in left if self.counts[p.key] == low])
        self._last = choice.key
        return choice

    def record(self, prompt: Prompt) -> None:
        self.counts[prompt.key] += 1

    def skip(self, prompt: Prompt) -> None:
        self.skipped.add(prompt.key)
