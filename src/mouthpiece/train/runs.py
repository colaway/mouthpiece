"""Fine-tuning run directories, checkpoint files and activation of the `current` symlink.

    models/finetuned/
      20260101-120000/
        best.pth       state dict with the pretrained checkpoint's exact keys
        metrics.json   baseline / best val WER, per-epoch history, timing
        config.yaml    the full config the run used
      current -> 20260101-120000    only if that run beat the pretrained baseline
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from mouthpiece.config import Config
from mouthpiece.model.checkpoint import BEST_NAME, METRICS_NAME

CURRENT_LINK = "current"
CONFIG_NAME = "config.yaml"


def new_run_dir(output_dir: Path, now: datetime | None = None) -> Path:
    """Create `<output_dir>/<YYYYmmdd-HHMMSS>/` (suffixed if that second is taken)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    run, n = output_dir / stamp, 1
    while run.exists() or run.name == CURRENT_LINK:
        n += 1
        run = output_dir / f"{stamp}-{n}"
    run.mkdir()
    return run


def save_weights(model: nn.Module, path: Path) -> None:
    """Full state dict on CPU, keys unchanged, so `load_vsr_model(strict=True)` loads it."""
    sd = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save(sd, tmp)
    os.replace(tmp, path)


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / METRICS_NAME).write_text(json.dumps(metrics, indent=2) + "\n")


def config_to_dict(cfg: Config) -> dict[str, Any]:
    raw = asdict(cfg)
    raw.pop("root", None)
    out: dict[str, Any] = json.loads(json.dumps(raw))  # tuples -> lists for YAML
    return out


def write_config(run_dir: Path, cfg: Config) -> None:
    (run_dir / CONFIG_NAME).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


def should_activate(best_wer: float | None, baseline_wer: float | None, real_val: bool) -> bool:
    """Activate only a run that strictly beat the pretrained model on a real val split."""
    return (real_val and best_wer is not None and baseline_wer is not None
            and best_wer < baseline_wer)


def activate(run_dir: Path, link: Path | None = None) -> Path:
    """Point `<output_dir>/current` at `run_dir` (relative symlink, replaced atomically)."""
    if not (run_dir / BEST_NAME).is_file():
        raise FileNotFoundError(f"{run_dir / BEST_NAME} missing, refusing to activate")
    link = link or run_dir.parent / CURRENT_LINK
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f"{link} exists and is not a symlink; not replacing it")
    target = os.path.relpath(run_dir.resolve(), link.parent.resolve())
    tmp = link.with_name(f".{link.name}.tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(target, target_is_directory=True)
    os.replace(tmp, link)
    return link
