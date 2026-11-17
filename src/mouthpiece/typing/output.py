"""The output stage: `Commit` in, keystrokes / actions out."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mouthpiece.config import TypingConfig
from mouthpiece.typing.actions import AppSignal
from mouthpiece.typing.composer import Composer, Plan
from mouthpiece.typing.emitter import Defer, Emitter, run_steps


class CommitLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def mode(self) -> str: ...


class TypingOutput:
    """Callable handler for `MenuBarApp.committed`.

    `can_post` reports whether events will reach other apps (Accessibility); when it is
    False, nothing is typed, the composer state is left as it was and `on_warning` is told.
    Internal signals (pause/resume) need no permission.
    """

    def __init__(
        self,
        cfg: TypingConfig,
        emitter: Emitter,
        defer: Defer,
        can_post: Callable[[], bool | None],
        on_signal: Callable[[AppSignal], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg, self.emitter, self.defer = cfg, emitter, defer
        self.can_post = can_post
        self.on_signal, self.on_warning = on_signal, on_warning
        self.log = log or (lambda msg: print(msg, flush=True))
        self.composer = Composer(cfg.history)

    def __call__(self, commit: CommitLike) -> Plan:
        plan = self.composer.plan(commit.text, commit.mode)
        if plan.steps and not self.can_post():
            self.composer.rollback()
            msg = "typing needs Accessibility permission"
            self.log(f"mouthpiece: [{commit.mode}] {commit.text!r}: not typed — {msg}")
            if self.on_warning is not None:
                self.on_warning(msg)
            return Plan(note="not typed")
        self.log(f"mouthpiece: [{commit.mode}] {commit.text!r} → {plan.note}")
        if plan.signal is not None and self.on_signal is not None:
            self.on_signal(plan.signal)
        try:
            run_steps(plan.steps, self.emitter, self.defer)
        except Exception as exc:     # never take the app down over one utterance
            self.log(f"mouthpiece: typing failed: {exc}")
            if self.on_warning is not None:
                self.on_warning(f"typing failed: {exc}")
        return plan

    def user_input(self) -> None:
        """The user typed or clicked: the cursor may have moved."""
        self.composer.forget()
