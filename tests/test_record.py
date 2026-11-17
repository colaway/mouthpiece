from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MEAN_FACE, similarity, synthetic_face

from mouthpiece.cli import main
from mouthpiece.config import Config
from mouthpiece.metrics import corpus_wer, normalize_text, wer
from mouthpiece.preprocess.pipeline import ClipResult
from mouthpiece.record.manifest import (
    ManifestEntry,
    ManifestError,
    PersonalStore,
    assign_split,
    read_manifest,
)
from mouthpiece.record.prompts import (
    COMMAND_PHRASES,
    DICTATION_SENTENCES,
    Prompt,
    PromptScheduler,
    builtin_prompts,
    load_prompt_file,
)
from mouthpiece.record.report import check_dataset, compute_stats, format_stats
from mouthpiece.record.session import Phase, RecorderSession, Take, check_take, process_take
from mouthpiece.train.dataset import IGNORE_ID, PersonalDataset, collate

LM = np.zeros((68, 2), np.float32)


def fake_encode(text: str) -> list[int]:
    return [len(w) for w in normalize_text(text).split()]


def fake_clip(t: int = 20, seed: int = 0) -> ClipResult:
    rng = np.random.default_rng(seed)
    return ClipResult(rng.integers(0, 255, (t, 96, 96), dtype=np.uint8), [LM] * t, 25.0)


def frames(t: int) -> list[np.ndarray]:
    return [np.zeros((48, 64, 3), np.uint8) for _ in range(t)]


def make_store(tmp_path: Path, **kw: object) -> PersonalStore:
    return PersonalStore(tmp_path / "personal", tmp_path / "personal/manifest.jsonl", **kw)  # type: ignore[arg-type]


# --- metrics -----------------------------------------------------------------


def test_normalize_and_wer() -> None:
    assert normalize_text("Don’t  send it, yet!") == "DON'T SEND IT YET"
    assert wer("select all", "SELECT ALL") == 0.0
    assert wer("select all", "select") == 0.5
    assert wer("a b", "a x b c") == 1.0
    assert wer("", "") == 0.0 and wer("", "x") == 1.0
    assert corpus_wer([("a b", "a b"), ("c d", "c")]) == 0.25
    assert corpus_wer([]) == 0.0


# --- prompts -----------------------------------------------------------------


def test_builtin_prompts_are_clean() -> None:
    ps = builtin_prompts()
    assert len(ps) == len(COMMAND_PHRASES) + len(DICTATION_SENTENCES)
    keys = [p.key for p in ps]
    assert len(set(keys)) == len(keys) and all(keys)
    # LRS3-style text: what we show is what we score (no digits/punctuation to verbalize)
    assert all(normalize_text(p.text) == p.text.upper() for p in ps)
    assert {p.kind for p in builtin_prompts(["command"])} == {"command"}


def test_load_prompt_file(tmp_path: Path) -> None:
    f = tmp_path / "p.txt"
    f.write_text("# my phrases\nopen slack\n\n  ship it  # comment\n!!!\n")
    assert load_prompt_file(f) == [Prompt("open slack", "custom"), Prompt("ship it", "custom")]


def test_scheduler_fewest_first_cap_and_skip() -> None:
    ps = [Prompt("a", "command"), Prompt("b", "command"), Prompt("c", "command"),
          Prompt("A", "custom")]  # duplicate of "a" after normalization
    s = PromptScheduler(ps, existing={"A": 2, "B": 1}, takes_per_phrase=2, seed=0)
    assert len(s.prompts) == 3
    assert [p.text for p in s.remaining()] == ["b", "c"]
    first = s.next()
    assert first is not None and first.text == "c"   # 0 takes beats 1 take
    s.record(first)
    nxt = s.next()
    assert nxt is not None and nxt.text == "b"
    s.skip(nxt)
    assert s.next() is not None   # "c" still has 1 < 2 takes
    s.record(Prompt("c", "command"))
    assert s.next() is None


def test_scheduler_avoids_immediate_repeat() -> None:
    s = PromptScheduler([Prompt("a", "command"), Prompt("b", "command")], seed=1)
    seq = [s.next() for _ in range(6)]
    assert all(x != y for x, y in zip(seq, seq[1:], strict=False))


# --- manifest / store ---------------------------------------------------------


def test_store_round_trip_never_overwrites(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    now = datetime(2026, 1, 1, 10, 0, 0, 123000)
    e1 = store.save("Select all", "command", [1, 2], fake_clip(20), frames(20),
                    "SELECT ALL", 0.0, now=now)
    e2 = store.save("new line", "command", [3], fake_clip(30, 1), None, None, None, now=now)
    assert e1.id == "20260101-100000-123" and e2.id == "20260101-100000-123-1"
    d1, d2 = store.clip_dir(e1), store.clip_dir(e2)
    assert {p.name for p in d1.iterdir()} == {"mouth.npy", "mouth.mp4", "video.mp4", "meta.json"}
    assert not (d2 / "video.mp4").exists()
    assert read_manifest(store.manifest) == [e1, e2]
    assert store.phrase_counts() == {"SELECT ALL": 1, "NEW LINE": 1}
    assert e1.frames == 20 and e1.duration_s == pytest.approx(0.8)
    assert e1.created == "2026-01-01T10:00:00" and e1.path == f"clips/{e1.id}"
    assert np.array_equal(store.load_mouths(e1), fake_clip(20).mouths)
    lines = store.manifest.read_text().splitlines()
    assert json.loads(lines[0])["tokens"] == [1, 2] and len(lines) == 2


def test_manifest_extra_fields_and_errors(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    e = store.save("yes", "command", [1], fake_clip(), now=datetime(2026, 1, 1))
    d = json.loads(e.to_json())
    d["future_field"] = 7
    assert ManifestEntry.from_dict(d).extra == {"future_field": 7}
    with store.manifest.open("a") as f:
        f.write("{not json\n")
    with pytest.raises(ManifestError, match=":2:"):
        read_manifest(store.manifest)
    assert read_manifest(tmp_path / "missing.jsonl") == []


def test_assign_split_deterministic() -> None:
    ids = [f"clip{i}" for i in range(2000)]
    splits = [assign_split(i, 0.1) for i in ids]
    assert splits == [assign_split(i, 0.1) for i in ids]
    assert 0.07 < splits.count("val") / len(ids) < 0.13
    assert assign_split("x", 0.0) == "train"


# --- takes / session ----------------------------------------------------------


def test_check_take_rejections(cfg: Config) -> None:
    r = cfg.record
    assert "too short" in (check_take(cfg, frames(r.min_frames - 1), [LM]) or "")
    assert "too long" in (check_take(cfg, frames(r.max_frames + 1), [LM]) or "")
    lms = [LM] * 8 + [None] * 2
    assert "face found in only 80%" in (check_take(cfg, frames(20), lms) or "")
    assert check_take(cfg, frames(20), [LM] * 20) is None


@pytest.mark.skipif(not MEAN_FACE.exists(), reason="mean face not downloaded")
def test_process_take_crops_and_transcribes(cfg: Config, mean_face: np.ndarray) -> None:
    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    p = Prompt("select all", "command")
    take = process_take(cfg, p, [frame] * 20, [lm] * 20, lambda m: "SELECT HALL")
    assert take.ok and take.clip is not None and take.clip.mouths.shape == (20, 96, 96)
    assert take.hypothesis == "SELECT HALL" and take.wer == 0.5
    bad = process_take(cfg, p, [frame] * 3, [lm] * 3)
    assert not bad.ok and "too short" in (bad.error or "")


def _session(tmp_path: Path, target: int = 2, n_prompts: int = 3) -> RecorderSession:
    ps = [Prompt(f"phrase {w}", "command") for w in ("one", "two", "three")[:n_prompts]]
    return RecorderSession(PromptScheduler(ps, seed=0), make_store(tmp_path), fake_encode,
                           target)


def test_session_flow(tmp_path: Path) -> None:
    s = _session(tmp_path)
    assert s.phase is Phase.PROMPT and s.prompt is not None
    assert s.keep() is None and s.end_take() is None   # nothing to keep / not recording
    assert s.begin_take() and not s.begin_take()
    p = s.end_take()
    assert p == s.prompt and s.phase is Phase.PROCESSING
    s.submit(Take(p, frames(3), error="too short"))
    assert s.phase is Phase.PROMPT and s.stats.rejected == 1 and "too short" in (s.message or "")

    s.begin_take()
    s.end_take()
    s.submit(Take(Prompt("stale", "command"), frames(20), fake_clip()))   # ignored
    assert s.phase is Phase.PROCESSING
    s.submit(Take(p, frames(20), fake_clip(), hypothesis="PHRASE"))
    assert s.phase is Phase.REVIEW
    s.redo()
    assert s.phase is Phase.PROMPT and s.stats.redone == 1 and s.prompt == p

    s.begin_take()
    s.end_take()
    s.submit(Take(p, frames(20), fake_clip(), hypothesis="PHRASE"))
    assert s.begin_take() and s.stats.redone == 2   # recording over a review = redo
    s.end_take()
    s.submit(Take(p, frames(20), fake_clip(), hypothesis="PHRASE"))
    entry = s.keep()
    assert entry is not None and entry.text == p.text and entry.baseline == "PHRASE"
    assert entry.baseline_wer == 0.5 and entry.tokens == fake_encode(p.text)
    assert s.stats.kept == 1 and s.stats.baseline_wer == 0.5
    assert s.prompt != p and s.takes_of(p) == 1

    skipped = s.prompt
    assert skipped is not None
    s.skip()
    assert s.stats.skipped == 1 and s.prompt not in (skipped, None)
    q = s.prompt
    assert q is not None
    s.begin_take()
    s.end_take()
    s.submit(Take(q, frames(20), fake_clip()))
    s.keep()
    assert s.phase is Phase.DONE and s.prompt is None   # target of 2 reached
    assert not s.begin_take()
    assert len(read_manifest(s.store.manifest)) == 2


def test_session_done_when_prompts_exhausted(tmp_path: Path) -> None:
    s = _session(tmp_path, target=10, n_prompts=1)
    s.skip()
    assert s.phase is Phase.DONE


# --- dataset ------------------------------------------------------------------


def _populate(tmp_path: Path, cfg: Config) -> tuple[Config, PersonalStore]:
    cfg = replace(cfg, data=replace(cfg.data, personal_dir=str(tmp_path / "personal"),
                                    manifest=str(tmp_path / "personal/manifest.jsonl")))
    store = PersonalStore.from_config(cfg)
    store.val_fraction = 0.5
    for i, (text, t) in enumerate([("select all", 20), ("new line please", 35), ("yes", 14),
                                   ("scroll down", 25)]):
        store.save(text, "command", fake_encode(text), fake_clip(t, i), None, text.upper(),
                   0.0, now=datetime(2026, 1, 1, 0, 0, i))
    return cfg, store


def test_dataset_and_collate(tmp_path: Path, cfg: Config) -> None:
    cfg, store = _populate(tmp_path, cfg)
    ds = PersonalDataset.from_config(cfg)
    assert len(ds) == 4 and ds.total_frames == 94
    s = ds[1]
    assert s.video.shape == (35, 88, 88) and s.video.dtype == torch.float32
    assert s.tokens.tolist() == fake_encode("new line please") and s.text == "new line please"
    # same normalization as inference
    raw = store.load_mouths(ds.entries[1]).astype(np.float32)[:, 4:92, 4:92] / 255
    assert torch.allclose(s.video, torch.from_numpy((raw - 0.421) / 0.165), atol=1e-5)

    b = collate([ds[i] for i in range(4)])
    assert len(b) == 4 and b.videos.shape == (4, 35, 88, 88)
    assert b.video_lengths.tolist() == [20, 35, 14, 25]
    assert b.token_lengths.tolist() == [2, 3, 1, 2] and b.tokens.shape == (4, 3)
    assert b.tokens[2, 1:].tolist() == [IGNORE_ID, IGNORE_ID]
    assert torch.all(b.videos[2, 14:] == 0)
    assert b.to("cpu").ids == b.ids
    with pytest.raises(ValueError):
        collate([])

    train = PersonalDataset.from_config(cfg, "train")
    val = PersonalDataset.from_config(cfg, "val")
    assert len(train) + len(val) == 4
    assert {e.id for e in train.entries}.isdisjoint(e.id for e in val.entries)

    flipped = PersonalDataset.from_config(cfg, transform=lambda m: m[:, :, ::-1])
    assert not torch.equal(flipped[0].video, ds[0].video)


# --- stats / check / cli ------------------------------------------------------


def test_stats_and_check(tmp_path: Path, cfg: Config) -> None:
    cfg, store = _populate(tmp_path, cfg)
    entries = store.entries()
    st = compute_stats(entries, rescore=lambda e: "")
    assert st.clips == 4 and st.minutes == pytest.approx(94 / 25 / 60)
    assert st.baseline_wer["all"] == 0.0 and st.rescored_wer["all"] == 1.0
    assert st.by_kind == {"command": 4}
    text = format_stats(st, target=5)
    assert "clips      4" in text and "4 recorded phrase(s) below 5 takes" in text

    errors, warnings = check_dataset(store.root, entries, fake_encode)
    assert errors == [] and warnings == []
    np.save(store.clip_dir(entries[0]) / "mouth.npy", np.zeros((3, 96, 96), np.uint8))
    (store.clip_dir(entries[1]) / "mouth.npy").unlink()
    (store.root / "clips/orphan").mkdir()
    errors, warnings = check_dataset(store.root, entries, lambda t: [0])
    assert any("3 frames on disk" in e for e in errors)
    assert any("missing" in e for e in errors)
    assert any("orphan" in w for w in warnings)
    assert any("tokens differ" in w for w in warnings)


def test_dataset_cli(tmp_path: Path, cfg: Config, capsys: pytest.CaptureFixture[str]) -> None:
    conf = tmp_path / "c.yaml"
    conf.write_text(f"data:\n  personal_dir: {tmp_path}/personal\n"
                    f"  manifest: {tmp_path}/personal/manifest.jsonl\n")
    assert main(["--config", str(conf), "dataset", "stats"]) == 0
    assert "No clips yet" in capsys.readouterr().out
    _populate(tmp_path, cfg)
    assert main(["--config", str(conf), "dataset", "stats"]) == 0
    assert "clips      4" in capsys.readouterr().out
    assert main(["--config", str(conf), "dataset", "check"]) == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_record_config_validation(tmp_path: Path) -> None:
    from mouthpiece.config import ConfigError, load_config

    p = tmp_path / "c.yaml"
    p.write_text("record:\n  min_frames: 50\n  max_frames: 10\n")
    with pytest.raises(ConfigError, match="min_frames"):
        load_config(p)
    p.write_text("record:\n  val_fraction: 1.5\n")
    with pytest.raises(ConfigError, match="val_fraction"):
        load_config(p)


# --- window -------------------------------------------------------------------


@pytest.mark.skipif(not MEAN_FACE.exists(), reason="mean face not downloaded")
def test_recorder_window_offscreen(tmp_path: Path, cfg: Config, mean_face: np.ndarray) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    from mouthpiece.app.recorder_window import RecorderWindow
    from mouthpiece.preprocess.preview_engine import PreviewEngine

    app = QApplication.instance() or QApplication([])
    session = _session(tmp_path)
    win = RecorderWindow(cfg, session)
    sent: list[str] = []
    win._send = sent.append  # type: ignore[method-assign]

    def key(k: Qt.Key, press: bool = True) -> QKeyEvent:
        t = QEvent.Type.KeyPress if press else QEvent.Type.KeyRelease
        return QKeyEvent(t, k, Qt.KeyboardModifier.NoModifier)

    prompt = session.prompt
    assert prompt is not None and win.phrase_label.text() == prompt.text
    assert "take 1 / 2" in win.progress_label.text()
    assert win.eventFilter(win, key(Qt.Key.Key_Space))
    assert session.phase is Phase.RECORDING and "recording" in win.take_label.text()
    assert win.eventFilter(win, key(Qt.Key.Key_Space, press=False))
    assert sent == ["ptt_start", "ptt_stop"]
    win.on_clip(frames(20), [LM] * 20)
    assert session.phase is Phase.PROCESSING and win.worker_takes.jobs.qsize() == 1
    win.on_take(Take(prompt, frames(20), fake_clip(), hypothesis="PHRASE"))
    assert session.phase is Phase.REVIEW and "model heard" in win.take_label.text()
    assert win.eventFilter(win, key(Qt.Key.Key_Return))
    assert session.stats.kept == 1 and "kept" in win.session_labels["kept"].text()
    assert win.eventFilter(win, key(Qt.Key.Key_N))
    assert session.stats.skipped == 1

    frame, lm = synthetic_face(mean_face, similarity(1.2, 0, 180, 60))
    eng = PreviewEngine(cfg, lambda f, t: lm, out_root=tmp_path)
    win.update_state(eng.process(frame, 0.0))
    win.show()
    app.processEvents()
    assert win.grab().toImage().width() > 0
    win.close()
