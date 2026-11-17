"""Non-GUI logic of the menu-bar app (Phase 5): hotkey, push-to-talk, picker, commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mouthpiece.app.controller import (
    PickerModel,
    PickerResult,
    alternatives,
    load_command_phrases,
    mode_switch,
    uses_picker,
)
from mouthpiece.app.hotkey import (
    FLAG_COMMAND,
    FLAGS_CHANGED,
    HOTKEYS,
    KEY_DOWN,
    KEY_UP,
    KeyEvent,
    PttAction,
    PushToTalk,
    classify,
    get_hotkey,
    picker_key,
)
from mouthpiece.config import Config, ConfigError, load_config
from mouthpiece.decode.recognizer import NBestEntry, Transcript
from mouthpiece.doctor import Status, check_accessibility, check_input_monitoring
from mouthpiece.record.prompts import COMMAND_PHRASES

ROPT = HOTKEYS["right_option"]


# --- hotkey classification ---------------------------------------------------


def test_right_option_down_up_uses_device_bit() -> None:
    # Right option down: generic option flag 0x80000 + right-device bit 0x40.
    assert classify(ROPT, FLAGS_CHANGED, 61, 0x80040) is KeyEvent.DOWN
    assert classify(ROPT, FLAGS_CHANGED, 61, 0x100) is KeyEvent.UP
    # Released right option while left option is still held: generic flag set, device bit not.
    assert classify(ROPT, FLAGS_CHANGED, 61, 0x80020) is KeyEvent.UP


def test_other_keys_interrupt() -> None:
    assert classify(ROPT, FLAGS_CHANGED, 58, 0x80020) is KeyEvent.OTHER  # left option
    assert classify(ROPT, KEY_DOWN, 14, 0x80040) is KeyEvent.OTHER       # Option+E
    assert classify(ROPT, KEY_UP, 14, 0) is None


def test_function_key_hotkey() -> None:
    f13 = get_hotkey("f13")
    assert not f13.is_modifier
    assert classify(f13, KEY_DOWN, 105, 0) is KeyEvent.DOWN
    assert classify(f13, KEY_DOWN, 105, 0, autorepeat=True) is None
    assert classify(f13, KEY_UP, 105, 0) is KeyEvent.UP
    assert classify(f13, FLAGS_CHANGED, 56, 0x20002) is KeyEvent.OTHER


def test_unknown_hotkey() -> None:
    with pytest.raises(ValueError, match="unknown hotkey"):
        get_hotkey("caps_lock")


def test_picker_key_ignores_shortcuts() -> None:
    assert picker_key(KEY_DOWN, 18, 0) == "1"
    assert picker_key(KEY_DOWN, 85, 0x200000) == "3"   # keypad (numpad flag is fine)
    assert picker_key(KEY_DOWN, 53, 0) == "escape"
    assert picker_key(KEY_DOWN, 18, FLAG_COMMAND) is None  # ⌘1 belongs to the focused app
    assert picker_key(KEY_UP, 18, 0) is None
    assert picker_key(KEY_DOWN, 0, 0) is None


# --- push-to-talk -------------------------------------------------------------


def test_hold_then_release_transcribes() -> None:
    ptt = PushToTalk(min_hold_s=0.3)
    assert ptt.press(0.0) is PttAction.START and ptt.recording
    assert ptt.press(0.1) is None                      # key repeat / duplicate down
    assert ptt.release(1.0) is PttAction.STOP and not ptt.recording
    assert ptt.release(1.1) is None


def test_short_tap_cancels() -> None:
    ptt = PushToTalk(min_hold_s=0.3)
    ptt.press(0.0)
    assert ptt.release(0.2) is PttAction.CANCEL


def test_other_key_cancels_and_release_is_silent() -> None:
    ptt = PushToTalk()
    ptt.press(0.0)
    assert ptt.interrupt() is PttAction.CANCEL
    assert ptt.interrupt() is None
    assert ptt.release(2.0) is None
    assert ptt.interrupt() is None                     # not held: nothing to cancel


def test_max_hold_stops_once() -> None:
    ptt = PushToTalk(max_hold_s=5.0)
    ptt.press(0.0)
    assert ptt.tick(4.9) is None
    assert ptt.tick(5.0) is PttAction.STOP
    assert ptt.tick(6.0) is None and ptt.release(7.0) is None
    assert ptt.press(8.0) is PttAction.START           # usable again


def test_reset_drops_current_hold() -> None:
    ptt = PushToTalk()
    ptt.press(0.0)
    ptt.reset()
    assert not ptt.recording and ptt.release(1.0) is None


# --- picker -------------------------------------------------------------------


def test_picker_digits_return_escape() -> None:
    m = PickerModel(["copy that", "cut that", "yes"])
    assert m.handle_key("2") == (True, PickerResult(1))
    assert m.handle_key("4") == (False, None)          # out of range: not swallowed
    assert m.handle_key("return") == (True, PickerResult(0))
    assert m.handle_key("escape") == (True, PickerResult(None))
    assert m.handle_key("x") == (False, None)


def test_picker_arrows_stop_auto_accept() -> None:
    m = PickerModel(["a", "b", "c"])
    assert m.timeout() == PickerResult(0, auto=True)
    assert m.handle_key("up") == (True, None)
    assert m.selected == 2 and m.touched
    assert m.timeout() is None
    m.handle_key("down")
    assert m.handle_key("return") == (True, PickerResult(0))


def test_picker_commit() -> None:
    m = PickerModel(["copy that", "cut that"])
    c = m.commit(PickerResult(1), "command")
    assert c is not None and c.text == "cut that" and c.index == 1 and not c.auto
    assert c.alternatives == ("copy that", "cut that") and c.mode == "command"
    assert m.commit(PickerResult(None), "command") is None
    with pytest.raises(ValueError):
        PickerModel([])


def _t(*texts: str) -> Transcript:
    return Transcript(texts[0] if texts else "",
                      [NBestEntry(t, -float(i), {}) for i, t in enumerate(texts)], 25, "cpu")


def test_alternatives_dedupe_and_limit() -> None:
    assert alternatives(_t("HELLO THERE", "hello there", "", "HELLO", "YES")) == (
        "HELLO THERE", "HELLO", "YES")
    assert alternatives(_t("a", "b", "c", "d"), limit=2) == ("a", "b")
    assert alternatives(_t("")) == ()


# --- commands / modes / config -----------------------------------------------


def test_mode_switch() -> None:
    assert mode_switch("Dictation mode") == "dictation"
    assert mode_switch("COMMAND MODE") == "command"
    assert mode_switch("copy that") is None


def test_command_phrases_with_user_file(cfg: Config, tmp_path: Path) -> None:
    assert load_command_phrases(cfg) == list(COMMAND_PHRASES)
    extra = tmp_path / "commands.txt"
    extra.write_text("# mine\nopen spotify\nCopy that  # duplicate of a built-in\n\n")
    phrases = load_command_phrases(replace(cfg, app=replace(cfg.app, commands=str(extra))))
    assert phrases[-1] == "open spotify"
    assert len(phrases) == len(COMMAND_PHRASES) + 1


def test_app_config_defaults_and_validation(tmp_path: Path) -> None:
    a = load_config().app
    assert a.hotkey == "right_option" and a.mode in ("command", "dictation")
    assert a.picker_timeout_s >= 0 and a.min_hold_s >= 0
    for bad, match in (("hotkey: caps_lock", "app.hotkey"), ("mode: shout", "app.mode"),
                       ("picker_timeout_s: -1", ">= 0"), ("typing: true", "unknown keys")):
        p = tmp_path / "c.yaml"
        p.write_text(f"app:\n  {bad}\n")
        with pytest.raises(ConfigError, match=match):
            load_config(p)


def test_doctor_permission_checks() -> None:
    assert check_input_monitoring(lambda: True).status is Status.OK
    r = check_input_monitoring(lambda: False)
    assert r.status is Status.WARN and "Input Monitoring" in r.fix
    assert check_input_monitoring(lambda: None).status is Status.WARN
    assert check_accessibility(lambda: True).status is Status.OK
    assert check_accessibility(lambda: False).status is Status.WARN


def test_dictation_skips_the_picker_unless_enabled() -> None:
    assert uses_picker("command", False) and uses_picker("command", True)
    assert not uses_picker("dictation", False)
    assert uses_picker("dictation", True)
    assert Config().app.dictation_picker is False
    assert load_config().app.dictation_picker is False
