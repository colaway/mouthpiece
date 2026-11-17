"""`mouthpiece dataset stats` / `check`: summaries and integrity checks of the personal data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mouthpiece.metrics import corpus_wer
from mouthpiece.record.manifest import CLIPS_DIR, ManifestEntry


@dataclass(frozen=True)
class Stats:
    clips: int
    minutes: float
    by_kind: dict[str, int]
    by_split: dict[str, int]
    by_phrase: list[tuple[str, int]]           # (text, takes), most-recorded first
    baseline_wer: dict[str, float | None]      # split -> stored baseline WER
    rescored_wer: dict[str, float | None] = field(default_factory=dict)


def _wer_by_split(pairs: Sequence[tuple[str, str, str]]) -> dict[str, float | None]:
    """(split, ref, hyp) -> WER per split and over all."""
    out: dict[str, float | None] = {}
    for name in ("train", "val", "all"):
        sel = [(r, h) for s, r, h in pairs if name == "all" or s == name]
        out[name] = corpus_wer(sel) if sel else None
    return out


def compute_stats(
    entries: Sequence[ManifestEntry],
    rescore: Callable[[ManifestEntry], str] | None = None,
) -> Stats:
    texts: dict[str, str] = {}
    phrase: Counter[str] = Counter()
    for e in entries:
        texts.setdefault(e.key, e.text)
        phrase[e.key] += 1
    stored = [(e.split, e.text, e.baseline) for e in entries if e.baseline is not None]
    rescored = {}
    if rescore is not None:
        rescored = _wer_by_split([(e.split, e.text, rescore(e)) for e in entries])
    return Stats(
        clips=len(entries),
        minutes=sum(e.duration_s for e in entries) / 60.0,
        by_kind=dict(Counter(e.kind for e in entries)),
        by_split=dict(Counter(e.split for e in entries)),
        by_phrase=[(texts[k], n) for k, n in phrase.most_common()],
        baseline_wer=_wer_by_split(stored),
        rescored_wer=rescored,
    )


def _fmt_wer(w: dict[str, float | None]) -> str:
    return "  ".join(f"{k} {'--' if v is None else f'{v:.1%}'}" for k, v in w.items())


def format_stats(s: Stats, top: int = 15, target: int | None = None) -> str:
    lines = [
        f"clips      {s.clips}  ({s.minutes:.1f} min)",
        "kinds      " + ("  ".join(f"{k} {n}" for k, n in sorted(s.by_kind.items())) or "--"),
        "split      " + ("  ".join(f"{k} {n}" for k, n in sorted(s.by_split.items())) or "--"),
        f"phrases    {len(s.by_phrase)} distinct",
        "baseline   " + _fmt_wer(s.baseline_wer) + "   (stored at record time)",
    ]
    if s.rescored_wer:
        lines.append("rescored   " + _fmt_wer(s.rescored_wer) + "   (current model)")
    if s.by_phrase:
        lines += ["", f"takes per phrase (top {min(top, len(s.by_phrase))}):"]
        lines += [f"  {n:3d}  {t}" for t, n in s.by_phrase[:top]]
        if target is not None:
            under = sum(n < target for _, n in s.by_phrase)
            lines.append(f"  … {under} recorded phrase(s) below {target} takes")
    return "\n".join(lines)


def check_dataset(
    root: Path,
    entries: Sequence[ManifestEntry],
    encode: Callable[[str], list[int]] | None = None,
    roi_size: int = 96,
) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors make the dataset unusable for training as-is."""
    errors: list[str] = []
    warnings: list[str] = []
    ids: Counter[str] = Counter(e.id for e in entries)
    errors += [f"duplicate id {i} ({n}x)" for i, n in ids.items() if n > 1]
    for e in entries:
        d = root / e.path
        npy = d / "mouth.npy"
        if not npy.exists():
            errors.append(f"{e.id}: missing {npy}")
            continue
        try:
            arr = np.load(npy, allow_pickle=False, mmap_mode="r")
        except (ValueError, OSError) as exc:
            errors.append(f"{e.id}: unreadable mouth.npy: {exc}")
            continue
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[1:] != (roi_size, roi_size):
            errors.append(f"{e.id}: mouth.npy is {arr.dtype} {arr.shape}, "
                          f"expected uint8 (T, {roi_size}, {roi_size})")
        elif arr.shape[0] != e.frames:
            errors.append(f"{e.id}: {arr.shape[0]} frames on disk, manifest says {e.frames}")
        if not e.tokens:
            errors.append(f"{e.id}: no target tokens")
        elif encode is not None and encode(e.text) != e.tokens:
            warnings.append(f"{e.id}: tokens differ from re-encoding {e.text!r}")
        meta = d / "meta.json"
        if not meta.exists():
            warnings.append(f"{e.id}: missing meta.json")
        else:
            try:
                json.loads(meta.read_text())
            except ValueError:
                warnings.append(f"{e.id}: meta.json is not valid JSON")
    known = {(root / e.path).resolve() for e in entries}
    clips = root / CLIPS_DIR
    if clips.is_dir():
        orphans = [p for p in sorted(clips.iterdir()) if p.is_dir() and p.resolve() not in known]
        warnings += [f"orphan clip folder (not in manifest): {p}" for p in orphans]
    return errors, warnings
