"""Which VSR weights inference uses: the activated fine-tuned run, else the pretrained model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mouthpiece.config import Config

BEST_NAME = "best.pth"
METRICS_NAME = "metrics.json"


@dataclass(frozen=True)
class ActiveCheckpoint:
    path: Path
    finetuned: bool
    run_dir: Path | None = None    # the fine-tuned run directory (symlink resolved)

    def describe(self, cfg: Config) -> str:
        if not self.finetuned or self.run_dir is None:
            return f"pretrained {_rel(cfg, self.path)}"
        text = f"fine-tuned {_rel(cfg, self.run_dir)}"
        m = read_metrics(self.run_dir)
        if m and isinstance(m.get("best_wer"), float) and isinstance(m.get("baseline_wer"), float):
            text += f" (val WER {m['best_wer']:.1%} vs pretrained {m['baseline_wer']:.1%})"
        return text


def _rel(cfg: Config, path: Path) -> str:
    try:
        return str(path.relative_to(cfg.root))
    except ValueError:
        return str(path)


def finetuned_weights(cfg: Config) -> Path | None:
    """`model.finetuned` as a weights file, if it is set and exists (a run dir -> best.pth)."""
    if not cfg.model.finetuned:
        return None
    p = cfg.resolve(cfg.model.finetuned)
    if p.is_dir():
        p = p / BEST_NAME
    return p if p.is_file() else None


def resolve_checkpoint(cfg: Config) -> ActiveCheckpoint:
    """Fine-tuned weights when available, else `model.checkpoint` (the pretrained model)."""
    ft = finetuned_weights(cfg)
    if ft is not None:
        return ActiveCheckpoint(ft, True, ft.resolve().parent)
    return ActiveCheckpoint(cfg.resolve(cfg.model.checkpoint), False)


def read_metrics(run_dir: Path) -> dict[str, Any] | None:
    try:
        raw: Any = json.loads((run_dir / METRICS_NAME).read_text())
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None
