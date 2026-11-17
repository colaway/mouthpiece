"""Personal dataset on disk: one folder per clip plus an append-only `manifest.jsonl`.

    data/personal/
      manifest.jsonl              one JSON object per kept clip (never rewritten)
      clips/<id>/mouth.npy        uint8 (T, 96, 96) mouth crops
      clips/<id>/mouth.mp4        preview of the crops
      clips/<id>/video.mp4        full camera frames (optional)
      clips/<id>/meta.json        frames, fps, duration, detection rate
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from mouthpiece.config import Config
from mouthpiece.metrics import normalize_text
from mouthpiece.preprocess.pipeline import ClipResult, Frame, save_clip

Split = Literal["train", "val"]
CLIPS_DIR = "clips"


def assign_split(clip_id: str, val_fraction: float) -> Split:
    """Deterministic split from a hash of the clip id (stable across machines and runs)."""
    h = int.from_bytes(hashlib.sha256(clip_id.encode()).digest()[:8], "big") / 2.0**64
    return "val" if h < val_fraction else "train"


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    text: str
    kind: str
    tokens: list[int]
    frames: int
    fps: float
    detection_rate: float
    created: str                       # ISO 8601, local time
    split: Split
    path: str                          # clip folder, relative to the personal dir
    baseline: str | None = None        # model hypothesis at record time
    baseline_wer: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def key(self) -> str:
        return normalize_text(self.text)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifestEntry:
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        args = {k: v for k, v in d.items() if k in known}
        args["extra"] = {**args.get("extra", {}), **extra}
        if args.get("split") not in ("train", "val"):
            raise ValueError(f"bad split {args.get('split')!r}")
        return cls(**args)


class ManifestError(ValueError):
    pass


def read_manifest(path: Path) -> list[ManifestEntry]:
    if not path.exists():
        return []
    out: list[ManifestEntry] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(ManifestEntry.from_dict(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ManifestError(f"{path}:{n}: {exc}") from exc
    return out


def append_entry(path: Path, entry: ManifestEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


class PersonalStore:
    """Writes kept takes. Never overwrites a clip folder or rewrites the manifest."""

    def __init__(self, root: Path, manifest: Path, val_fraction: float = 0.1,
                 save_video: bool = True) -> None:
        self.root, self.manifest = root, manifest
        self.val_fraction, self.save_video = val_fraction, save_video

    @classmethod
    def from_config(cls, cfg: Config) -> PersonalStore:
        return cls(cfg.resolve(cfg.data.personal_dir), cfg.resolve(cfg.data.manifest),
                   cfg.record.val_fraction, cfg.record.save_video)

    def entries(self) -> list[ManifestEntry]:
        return read_manifest(self.manifest)

    def phrase_counts(self) -> Counter[str]:
        return Counter(e.key for e in self.entries())

    def _new_dir(self, now: datetime) -> tuple[str, Path]:
        base = now.strftime("%Y%m%d-%H%M%S-%f")[:-3]   # millisecond resolution
        clips = self.root / CLIPS_DIR
        clips.mkdir(parents=True, exist_ok=True)
        for n in range(1000):
            cid = base if n == 0 else f"{base}-{n}"
            d = clips / cid
            try:
                d.mkdir()   # atomic: fails if it exists, so nothing is ever overwritten
                return cid, d
            except FileExistsError:
                continue
        raise RuntimeError(f"could not create a unique clip folder under {clips}")

    def save(
        self,
        text: str,
        kind: str,
        tokens: Sequence[int],
        clip: ClipResult,
        frames: Sequence[Frame] | None = None,
        baseline: str | None = None,
        baseline_wer: float | None = None,
        now: datetime | None = None,
    ) -> ManifestEntry:
        now = now or datetime.now()
        cid, d = self._new_dir(now)
        save_clip(d, clip, frames if self.save_video else None)
        entry = ManifestEntry(
            id=cid, text=text, kind=kind, tokens=[int(t) for t in tokens],
            frames=int(clip.mouths.shape[0]), fps=float(clip.fps),
            detection_rate=round(clip.detection_rate, 3),
            created=now.isoformat(timespec="seconds"),
            split=assign_split(cid, self.val_fraction),
            path=f"{CLIPS_DIR}/{cid}",
            baseline=baseline,
            baseline_wer=None if baseline_wer is None else round(baseline_wer, 4),
        )
        append_entry(self.manifest, entry)
        return entry

    def clip_dir(self, entry: ManifestEntry) -> Path:
        return self.root / entry.path

    def load_mouths(self, entry: ManifestEntry) -> npt.NDArray[np.uint8]:
        arr: npt.NDArray[np.uint8] = np.load(self.clip_dir(entry) / "mouth.npy",
                                             allow_pickle=False)
        return arr
