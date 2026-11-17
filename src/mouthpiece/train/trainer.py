"""`mouthpiece train`: fine-tune the pretrained VSR model on the personal recordings.

Every run starts from the pretrained checkpoint (`model.checkpoint`), measures its val WER
first (the baseline), trains the top of the encoder plus decoder and CTC head, keeps the best
epoch by val WER, and activates the run (`models/finetuned/current`) only if it beat the
baseline on a real val split.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mouthpiece.config import Config
from mouthpiece.decode.recognizer import Recognizer
from mouthpiece.metrics import corpus_wer
from mouthpiece.model.checkpoint import BEST_NAME
from mouthpiece.model.device import resolve_device, synchronize
from mouthpiece.model.lm import TransformerLM, load_lm
from mouthpiece.model.tokenizer import Tokenizer
from mouthpiece.model.vsr import VSRModel, load_vsr_model
from mouthpiece.record.manifest import ManifestEntry, read_manifest
from mouthpiece.train.augment import TrainAugment
from mouthpiece.train.dataset import PersonalDataset, collate
from mouthpiece.train.freeze import apply_freeze, set_train_mode
from mouthpiece.train.loss import compute_loss
from mouthpiece.train.runs import (
    activate,
    new_run_dir,
    save_weights,
    should_activate,
    write_config,
    write_metrics,
)

Log = Callable[[str], None]

SMOKE_STEPS = 4          # optimizer steps per epoch in a --smoke run
SMOKE_VAL_CLIPS = 8      # val clips decoded in a --smoke run
SMOKE_BEAM = 10          # beam size for --smoke evaluation


class TrainError(RuntimeError):
    """The run cannot start (no data, too few clips, ...)."""


@dataclass(frozen=True)
class TrainOptions:
    epochs: int | None = None      # override train.epochs
    force: bool = False            # allow a real run below train.min_clips
    smoke: bool = False            # tiny run to test the pipeline; never activated
    activate: bool = True          # activate the run if it beats the baseline
    device: str | None = None      # override model.device
    use_lm: bool = True            # decode val with the LM (like inference)


@dataclass
class EpochStats:
    epoch: int
    steps: int
    loss: float
    ctc: float
    att: float
    acc: float
    val_wer: float
    train_s: float
    eval_s: float
    lr: float

    def line(self) -> str:
        return (f"epoch {self.epoch:2d} ({self.steps} steps): loss {self.loss:7.2f} "
                f"(ctc {self.ctc:6.2f}, att {self.att:6.2f}, acc {self.acc:5.1%}), "
                f"val WER {self.val_wer:6.1%}, "
                f"train {self.train_s:5.1f}s, eval {self.eval_s:5.1f}s, lr {self.lr:.1e}")


@dataclass
class TrainResult:
    run_dir: Path
    baseline_wer: float
    best_wer: float | None
    best_epoch: int | None
    activated: bool
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataSplit:
    train: list[ManifestEntry]
    val: list[ManifestEntry]
    real_val: bool          # False when val fell back to the training clips
    warning: str | None = None


def split_entries(entries: Sequence[ManifestEntry]) -> DataSplit:
    """Manifest -> train/val. An empty side falls back to the other (with a warning)."""
    train = [e for e in entries if e.split == "train"]
    val = [e for e in entries if e.split == "val"]
    if not entries:
        raise TrainError("no clips in the manifest; record some with `mouthpiece record`")
    if not val:
        return DataSplit(train, train, False,
                         "no val clips: evaluating on the training clips (WER is optimistic, "
                         "the run will not be activated)")
    if not train:
        return DataSplit(val, val, False,
                         "no train clips: training on the val clips (the run will not be "
                         "activated)")
    return DataSplit(train, val, True)


def frame_batches(
    lengths: Sequence[int], max_frames: int, rng: np.random.Generator | None = None
) -> list[list[int]]:
    """Group clip indices by length so each padded batch has <= max_frames frames
    (batch size * longest clip). A clip longer than max_frames gets a batch of its own.
    Batch order is shuffled when `rng` is given."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1) * max(lengths[i], lengths[cur[-1]]) > max_frames:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1 over `warmup` steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: VSRModel, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW on trainable parameters; no weight decay on biases and norm weights."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)


@torch.no_grad()
def evaluate(rec: Recognizer, dataset: PersonalDataset) -> tuple[float, list[tuple[str, str]]]:
    """Corpus WER of `rec` over `dataset` (no augmentation), plus (ref, hyp) pairs."""
    rec.model.eval()
    pairs = []
    for i in range(len(dataset)):
        s = dataset[i]
        pairs.append((s.text, rec.transcribe(s.video.numpy()).text))
    return corpus_wer(pairs), pairs


def memory_mb(device: torch.device) -> float:
    if device.type == "mps":
        return float(torch.mps.driver_allocated_memory()) / 2**20
    return 0.0


def train_epoch(
    model: VSRModel,
    dataset: PersonalDataset,
    batches: list[list[int]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    frozen_blocks: int,
    cfg: Config,
    device: torch.device,
    max_steps: int | None = None,
) -> tuple[dict[str, float], int, float]:
    """One pass over `batches`; returns mean losses, steps taken, peak memory (MB)."""
    t = cfg.train
    set_train_mode(model, frozen_blocks)
    params = [p for p in model.parameters() if p.requires_grad]
    sums = {"loss": 0.0, "ctc": 0.0, "att": 0.0, "acc": 0.0}
    steps, peak = 0, 0.0
    for idx in batches[:max_steps] if max_steps else batches:
        batch = collate([dataset[i] for i in idx]).to(device)
        out = compute_loss(model, batch, frozen_blocks, t.ctc_weight, t.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(params, t.grad_clip)
        optimizer.step()
        scheduler.step()
        steps += 1
        for k, v in (("loss", float(out.loss.item())), ("ctc", out.ctc), ("att", out.att),
                     ("acc", out.acc)):
            sums[k] += v
        peak = max(peak, memory_mb(device))
    synchronize(device)
    return {k: v / max(1, steps) for k, v in sums.items()}, steps, peak


def _load_lm(cfg: Config, device: torch.device) -> TransformerLM | None:
    m = cfg.model
    if not m.language_model or cfg.decode.lm_weight <= 0:
        return None
    path = cfg.resolve(m.language_model)
    if not path.exists():
        return None
    conf = m.language_model_config or str(path.with_suffix(".json"))
    return load_lm(path, cfg.resolve(conf), device)


def run_training(cfg: Config, opts: TrainOptions, log: Log = print) -> TrainResult:
    t = cfg.train
    entries = read_manifest(cfg.resolve(cfg.data.manifest))
    if len(entries) < t.min_clips and not (opts.force or opts.smoke):
        raise TrainError(
            f"only {len(entries)} clips recorded, train.min_clips is {t.min_clips}. Record "
            "more with `mouthpiece record`, use --smoke to test the pipeline, or --force.")
    split = split_entries(entries)
    if split.warning:
        log(f"⚠️  {split.warning}")
    val_entries = split.val[:SMOKE_VAL_CLIPS] if opts.smoke else split.val
    epochs = opts.epochs or (1 if opts.smoke else t.epochs)
    beam = t.eval_beam_size or cfg.decode.beam_size
    if opts.smoke:
        beam = min(beam, SMOKE_BEAM)

    device = resolve_device(opts.device or cfg.model.device)
    torch.manual_seed(t.seed)
    m, p = cfg.model, cfg.preprocess
    tok = Tokenizer.from_files(cfg.resolve(m.tokenizer_units), cfg.resolve(m.tokenizer))
    t0 = time.perf_counter()
    # Always start from the pretrained weights, never from a previous fine-tuned run.
    model = load_vsr_model(cfg.resolve(m.checkpoint), cfg.resolve(m.model_config), device,
                           tok.vocab_size)
    lm = _load_lm(cfg, device) if opts.use_lm else None
    load_s = time.perf_counter() - t0
    freeze = apply_freeze(model, t.trainable_blocks)
    rec = Recognizer(model, tok, replace(cfg.decode, beam_size=beam), lm, p.model_input_size,
                     p.normalize_mean, p.normalize_std)

    root = cfg.resolve(cfg.data.personal_dir)
    aug = TrainAugment.from_config(t, p.model_input_size, t.seed)
    train_ds = PersonalDataset(root, split.train, p.model_input_size, p.normalize_mean,
                               p.normalize_std, aug)
    val_ds = PersonalDataset(root, val_entries, p.model_input_size, p.normalize_mean,
                             p.normalize_std)
    log(f"device {device.type}, {len(train_ds)} train clips ({train_ds.total_frames} frames), "
        f"{len(val_ds)} val clips{'' if split.real_val else ' (= train)'}, "
        f"model + LM load {load_s:.1f}s")
    log(f"{freeze}; val decode beam {beam}, LM {'on' if lm is not None else 'off'}")

    run_dir = new_run_dir(cfg.resolve(t.output_dir))
    write_config(run_dir, cfg)
    log(f"run dir {run_dir}")

    t0 = time.perf_counter()
    baseline_wer, _ = evaluate(rec, val_ds)
    baseline_s = time.perf_counter() - t0
    log(f"pretrained baseline val WER {baseline_wer:.1%} ({baseline_s:.1f}s)")

    rng = np.random.default_rng(t.seed)
    lengths = [e.frames for e in split.train]
    steps_per_epoch = len(frame_batches(lengths, t.max_batch_frames))
    if opts.smoke:
        steps_per_epoch = min(steps_per_epoch, SMOKE_STEPS)
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(t.warmup_fraction * total))
    optimizer = build_optimizer(model, t.lr, t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, warmup, total))

    metrics: dict[str, Any] = {
        "smoke": opts.smoke, "device": device.type, "trainable_params": freeze.trainable,
        "total_params": freeze.total, "trainable_blocks": freeze.trainable_blocks,
        "train_clips": len(train_ds), "val_clips": len(val_ds), "real_val": split.real_val,
        "eval_beam_size": beam, "lm": lm is not None, "baseline_wer": baseline_wer,
        "best_wer": None, "best_epoch": None, "history": [], "peak_memory_mb": 0.0,
        "activated": False,
    }
    best: float | None = None
    best_epoch: int | None = None
    bad = 0
    try:
        for epoch in range(1, epochs + 1):
            batches = frame_batches(lengths, t.max_batch_frames, rng)
            t0 = time.perf_counter()
            losses, steps, peak = train_epoch(
                model, train_ds, batches, optimizer, scheduler, freeze.frozen_blocks, cfg,
                device, SMOKE_STEPS if opts.smoke else None)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            wer, _ = evaluate(rec, val_ds)
            stats = EpochStats(epoch, steps, losses["loss"], losses["ctc"], losses["att"],
                               losses["acc"], wer, train_s, time.perf_counter() - t0,
                               float(scheduler.get_last_lr()[0]))
            metrics["history"].append(vars(stats))
            metrics["peak_memory_mb"] = max(metrics["peak_memory_mb"], peak)
            improved = best is None or wer < best
            log(stats.line() + (f", peak mem {peak / 1024:.1f} GB" if peak else "")
                + ("  *" if improved else ""))
            if improved:
                best, best_epoch, bad = wer, epoch, 0
                save_weights(model, run_dir / BEST_NAME)
                metrics.update(best_wer=best, best_epoch=best_epoch)
            else:
                bad += 1
            write_metrics(run_dir, metrics)
            if bad >= t.patience:
                log(f"no val improvement for {bad} epochs, stopping early")
                break
    except KeyboardInterrupt:
        log("interrupted; keeping the best epoch so far")
        metrics["interrupted"] = True

    activated = False
    if opts.smoke:
        log("smoke run: not activating")
    elif not opts.activate:
        log("--no-activate: not activating")
    elif should_activate(best, baseline_wer, split.real_val):
        link = activate(run_dir)
        activated = True
        log(f"✅ val WER {best:.1%} beats pretrained {baseline_wer:.1%}: {link} -> {run_dir.name}")
    elif not split.real_val:
        log("no real val split: not activating")
    else:
        log(f"val WER {best if best is None else f'{best:.1%}'} did not beat pretrained "
            f"{baseline_wer:.1%}: not activating (inference keeps its current weights)")
    metrics["activated"] = activated
    write_metrics(run_dir, metrics)
    return TrainResult(run_dir, baseline_wer, best, best_epoch, activated, metrics)
