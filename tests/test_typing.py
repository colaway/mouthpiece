"""Phase 6 typing output: key combos, command actions, formatting, composer, emitter logic."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from mouthpiece.config import Config, ConfigError, TypingConfig, load_config
from mouthpiece.metrics import normalize_text
from mouthpiece.record.prompts import COMMAND_PHRASES
from mouthpiece.typing.actions import (
    COMMAND_ACTIONS,
    Ignore,
    Launch,
    Media,
    Press,
    Scroll,
    TypeText,
    Wait,
    action_for,
)
from mouthpiece.typing.composer import Composer, format_words
from mouthpiece.typing.emitter import (
    RecordingEmitter,
    launch_command,
    run_steps,
    use_paste,
    utf16_chunks,
)
from mouthpiece.typing.keys import BACKSPACE, MODIFIER_FLAGS, combo
from mouthpiece.typing.output import TypingOutput

macos = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@dataclass(frozen=True)
class C:          # minimal Commit
    text: str
    mode: str


def now(seconds: float, fn: Any) -> None:
    fn()


def drive(*utterances: str | tuple[str, str]) -> RecordingEmitter:
    """Run utterances (a bare string: command if it is a known phrase, else dictation)."""
    em = RecordingEmitter()
    out = TypingOutput(TypingConfig(), em, now, can_post=lambda: True, log=lambda m: None)
    for u in utterances:
        if isinstance(u, tuple):
            out(C(*u))
        else:
            mode = "command" if normalize_text(u) in COMMAND_ACTIONS else "dictation"
            out(C(u, mode))
    return em


# --- keys ----------------------------------------------------------------------


def test_combo_parsing_canonicalises_modifiers() -> None:
    c = combo("Shift+CMD+z")
    assert (c.key, c.mods, c.keycode) == ("z", ("shift", "cmd"), 6)
    assert c.flags == MODIFIER_FLAGS["shift"] | MODIFIER_FLAGS["cmd"]
    assert combo("option+backspace").mods == ("opt",)
    assert combo("cmd+=") == combo("command+plus")
    assert str(combo("cmd+shift+4")) == "⇧⌘4"


def test_combo_arrow_keys_carry_fn_and_numpad_flags() -> None:
    assert combo("left").flags & MODIFIER_FLAGS["fn"]
    assert combo("cmd+left").flags & MODIFIER_FLAGS["cmd"]
    assert combo("a").flags == 0


@pytest.mark.parametrize("bad", ["", "cmd+", "hyper+a", "cmd+nope"])
def test_combo_rejects_bad_specs(bad: str) -> None:
    with pytest.raises(ValueError):
        combo(bad)


# --- command table -------------------------------------------------------------


def test_every_builtin_command_phrase_has_an_action() -> None:
    missing = [p for p in COMMAND_PHRASES if normalize_text(p) not in COMMAND_ACTIONS]
    assert missing == []
    ignored = {p for p in COMMAND_PHRASES if isinstance(action_for(p), Ignore)}
    assert ignored == {"dictation mode", "command mode"}   # the app consumes these itself


def test_action_lookup_is_normalised_and_unknown_phrases_are_ignored() -> None:
    assert action_for("Copy That!") == action_for("copy that")
    assert isinstance(action_for("open the pod bay doors"), Ignore)


def test_media_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        Media("eject")


# --- formatting ----------------------------------------------------------------


@pytest.mark.parametrize("raw,cap,casing,want", [
    ("I THINK I'M READY", True, None, "I think I'm ready"),
    ("AND I'LL SEE IF I'VE GOT IT", False, None, "and I'll see if I've got it"),
    ("IT IS IN THE KIT", True, None, "It is in the kit"),     # no stray "I" inside words
    ("  HELLO   THERE ", True, None, "Hello there"),
    ("HELLO", False, "capital", "Hello"),
    ("HELLO", True, "lower", "hello"),
    ("I AGREE", True, "lower", "I agree"),
    ("hello there", True, "upper", "HELLO THERE"),
    ("DON’T", True, None, "Don't"),
    ("   ", True, None, ""),
])
def test_format_words(raw: str, cap: bool, casing: Any, want: str) -> None:
    assert format_words(raw, cap, casing) == want


# --- composer ------------------------------------------------------------------


def test_dictation_spacing_and_punctuation() -> None:
    em = drive("I THINK SO", "comma", "BUT LET ME CHECK", "period", "HOW ARE YOU",
               "question mark")
    assert em.buffer == "I think so, but let me check. How are you?"


def test_no_auto_period_between_utterances() -> None:
    assert drive("HELLO THERE", "HOW ARE YOU").buffer == "Hello there how are you"


def test_new_line_and_paragraph_start_sentences_without_leading_space() -> None:
    em = drive("FIRST LINE", "new line", "SECOND", "new paragraph", "THIRD")
    assert em.buffer == "First line\nSecond\n\nThird"
    assert "press return" in em.calls


def test_quotes_and_brackets_attach_correctly() -> None:
    em = drive("HE SAID", "open quote", "HI THERE", "close quote", "AND LEFT",
               "open bracket", "QUIETLY", "close bracket")
    assert em.buffer == 'He said "hi there" and left (quietly)'


def test_casing_modifiers_apply_to_the_next_utterance_only() -> None:
    em = drive("HELLO", "all caps", "WORLD", "AGAIN", "period", "no caps", "SMALL",
               "capital letter", "BIG")
    assert em.buffer == "Hello WORLD again. small Big"


def test_courtesy_phrases_are_typed_as_text() -> None:
    assert drive("thank you", "comma", "sounds good").buffer == "Thank you, sounds good"


def test_scratch_that_removes_the_last_chunk_and_restores_state() -> None:
    em = drive("I WILL BE LATE", "period", "SORRY", "scratch that", "scratch that",
               "AND SORRY")
    assert em.buffer == "I will be late and sorry"


def test_scratch_that_with_nothing_typed_does_nothing() -> None:
    em = drive("scratch that")
    assert em.calls == [] and em.buffer == ""


def test_delete_that_falls_back_to_one_backspace() -> None:
    em = drive("select the word", "delete that")
    assert em.calls[-1] == "press backspace"


def test_navigation_forgets_history_but_copy_keeps_it() -> None:
    assert drive("HELLO", "copy that", "scratch that").buffer == ""
    em = drive("HELLO", "move left", "scratch that")
    assert em.buffer == "Hello" and em.calls[-1] == "press left"


def test_user_input_forgets_history() -> None:
    em = RecordingEmitter()
    out = TypingOutput(TypingConfig(), em, now, can_post=lambda: True, log=lambda m: None)
    out(C("HELLO", "dictation"))
    out.user_input()
    out(C("scratch that", "command"))
    assert em.buffer == "Hello"


def test_history_is_bounded() -> None:
    comp = Composer(history=2)
    for w in ("ONE", "TWO", "THREE"):
        comp.plan(w, "dictation")
    assert len(comp.history) == 2
    comp.plan("scratch that", "command")
    comp.plan("scratch that", "command")
    assert comp.plan("scratch that", "command").steps == ()


def test_key_commands_map_to_combos() -> None:
    em = drive("select all", "copy that", "undo that", "redo that", "select the line",
               "next tab", "zoom in")
    assert em.calls == ["press ⌘a", "press ⌘c", "press ⌘z", "press ⇧⌘z", "press ⌘left",
                        "press ⇧⌘right", "press ⌃tab", "press ⌘="]


def test_system_commands() -> None:
    em = drive("volume up", "mute", "play music", "scroll down", "open the terminal",
               "search the web", "take a screenshot")
    assert em.calls == ["media volume_up", "media mute", "media play_pause", "scroll -5",
                        "launch app:Terminal", "launch default:https", "press ⌘t",
                        "press ⇧⌘4"]


def test_rollback_restores_state() -> None:
    comp = Composer()
    comp.plan("HELLO", "dictation")
    before = (comp.state, list(comp.history))
    comp.plan("WORLD", "dictation")
    comp.rollback()
    assert (comp.state, list(comp.history)) == before


# --- output stage --------------------------------------------------------------


def test_without_accessibility_nothing_is_typed_and_state_is_kept() -> None:
    em = RecordingEmitter()
    warnings: list[str] = []
    granted = False
    out = TypingOutput(TypingConfig(), em, now, can_post=lambda: granted,
                       on_warning=warnings.append, log=lambda m: None)
    out(C("HELLO", "dictation"))
    assert em.calls == [] and warnings == ["typing needs Accessibility permission"]
    granted = True
    out(C("HELLO", "dictation"))
    assert em.buffer == "Hello"          # still capitalised: the failed commit was rolled back


def test_signals_need_no_permission() -> None:
    signals: list[str] = []
    out = TypingOutput(TypingConfig(), RecordingEmitter(), now, can_post=lambda: False,
                       on_signal=signals.append, log=lambda m: None)
    out(C("stop listening", "command"))
    out(C("start listening", "command"))
    assert signals == ["pause", "resume"]


def test_emitter_errors_are_reported_not_raised() -> None:
    class Broken(RecordingEmitter):
        def type_text(self, text: str) -> None:
            raise RuntimeError("boom")

    warnings: list[str] = []
    out = TypingOutput(TypingConfig(), Broken(), now, can_post=lambda: True,
                       on_warning=warnings.append, log=lambda m: None)
    out(C("HELLO", "dictation"))
    assert warnings == ["typing failed: boom"]


# --- emitter logic -------------------------------------------------------------


def test_run_steps_defers_after_wait() -> None:
    em = RecordingEmitter()
    pending: list[tuple[float, Any]] = []
    run_steps([Launch("app:X"), Wait(0.8), Press(combo("cmd+t")), Scroll(3),
               TypeText("hi"), Press(BACKSPACE, 2)], em,
              lambda s, fn: pending.append((s, fn)))
    assert em.calls == ["launch app:X"]
    (secs, fn), = pending
    assert secs == 0.8
    fn()
    assert em.calls[1:] == ["press ⌘t", "scroll +3", "type 'hi'", "press backspace",
                            "press backspace"]


def test_utf16_chunks_never_split_characters() -> None:
    assert utf16_chunks("a" * 45, 20) == ["a" * 20, "a" * 20, "a" * 5]
    chunks = utf16_chunks("x" * 19 + "😀" + "y", 20)
    assert chunks == ["x" * 19, "😀y"]
    assert all(len(c.encode("utf-16-le")) // 2 <= 20 for c in chunks)
    assert utf16_chunks("") == []


def test_use_paste() -> None:
    cfg = TypingConfig(paste_over_chars=10, paste_apps=("com.example.rdp",))
    assert not use_paste("short", cfg, "com.apple.TextEdit")
    assert use_paste("x" * 11, cfg, None)
    assert use_paste("short", cfg, "com.example.rdp")
    assert use_paste("short", replace(cfg, method="paste"), None)
    assert not use_paste("x" * 50, replace(cfg, method="keys"), "com.example.rdp")


def test_launch_command() -> None:
    assert launch_command("app:Terminal") == ["open", "-a", "Terminal"]
    assert launch_command("default:mailto", lambda s: "/Apps/Mail.app") == [
        "open", "/Apps/Mail.app"]
    assert launch_command("default:mailto", lambda s: None) is None
    assert launch_command("bogus") is None


# --- config --------------------------------------------------------------------


def test_typing_config_loads_from_repo_yaml() -> None:
    t = load_config().typing
    assert t.enabled and t.method == "auto" and isinstance(t.paste_apps, tuple)


@pytest.mark.parametrize("section,match", [
    ("typing:\n  method: shout\n", "typing.method"),
    ("typing:\n  history: 0\n", "history"),
    ("typing:\n  bogus: 1\n", "unknown keys"),
])
def test_typing_config_validation(tmp_path: Path, section: str, match: str) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(section)
    with pytest.raises(ConfigError, match=match):
        load_config(p)


def test_typing_paste_apps_list_becomes_tuple(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("typing:\n  paste_apps: [com.a, com.b]\n")
    assert load_config(p).typing.paste_apps == ("com.a", "com.b")
    assert Config().typing == TypingConfig()


# --- macOS event construction (built, never posted) -----------------------------


@macos
def test_quartz_events_carry_text_flags_and_tag() -> None:
    Q = pytest.importorskip("Quartz")
    from mouthpiece.app.hotkey import SYNTHETIC_TAG
    from mouthpiece.typing.emitter import QuartzEmitter

    em = QuartzEmitter(TypingConfig())
    evs = em.unicode_events("Héllo, wörld — 😀 " + "x" * 20)
    assert len(evs) % 2 == 0
    text = ""
    for ev in evs[::2]:
        n, s = Q.CGEventKeyboardGetUnicodeString(ev, 32, None, None)
        text += s[:n]
        assert Q.CGEventGetFlags(ev) & 0xFF0000 == 0   # no modifiers leak in
    assert text == "Héllo, wörld — 😀 " + "x" * 20
    c = combo("cmd+shift+z")
    ev = em.key_event(c.keycode, True, c.flags)
    assert Q.CGEventGetIntegerValueField(ev, Q.kCGKeyboardEventKeycode) == 6
    assert Q.CGEventGetFlags(ev) & c.flags == c.flags
    assert Q.CGEventGetIntegerValueField(ev, Q.kCGEventSourceUserData) == SYNTHETIC_TAG


@macos
def test_pasteboard_snapshot_and_restore_on_a_private_pasteboard() -> None:
    AppKit = pytest.importorskip("AppKit")
    from mouthpiece.typing.emitter import Pasteboard

    raw = AppKit.NSPasteboard.pasteboardWithUniqueName()
    try:
        raw.clearContents()
        raw.setString_forType_("original", AppKit.NSPasteboardTypeString)
        pb = Pasteboard(raw)
        saved = pb.snapshot()
        count = pb.set_text("typed by mouthpiece")
        assert raw.stringForType_(AppKit.NSPasteboardTypeString) == "typed by mouthpiece"
        assert pb.restore(saved, count)
        assert raw.stringForType_(AppKit.NSPasteboardTypeString) == "original"
        # The user copied something in between: leave their clipboard alone.
        count = pb.set_text("typed")
        raw.clearContents()
        raw.setString_forType_("user copy", AppKit.NSPasteboardTypeString)
        assert not pb.restore(saved, count)
        assert raw.stringForType_(AppKit.NSPasteboardTypeString) == "user copy"
    finally:
        raw.releaseGlobally()


def test_editing_phrases_work_as_commands_in_dictation_mode() -> None:
    em = drive(("I WILL BE LATE", "dictation"), ("COMMA", "dictation"),
               ("SORRY", "dictation"), ("scratch that", "dictation"),
               ("PERIOD", "dictation"), ("new line", "dictation"),
               ("SEE YOU", "dictation"), ("question mark", "dictation"))
    assert em.buffer == "I will be late,.\nSee you?"


def test_only_whole_utterances_become_commands_in_dictation() -> None:
    em = drive(("IT WAS A TRIAL PERIOD", "dictation"), ("copy that", "dictation"),
               ("thank you", "dictation"))
    assert em.buffer == "It was a trial period copy that thank you"
