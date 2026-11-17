"""Load and validate the single YAML config file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "mouthpiece.yaml"
CONFIG_ENV_VAR = "MOUTHPIECE_CONFIG"


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int = 0
    fps: int = 25
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class PreprocessConfig:
    face_landmarker_model: str = "models/preprocess/face_landmarker.task"
    mean_face: str = "models/preprocess/20words_mean_face.npy"
    stable_points: tuple[int, ...] = (28, 33, 36, 39, 42, 45, 48, 54)
    aligned_size: int = 256
    roi_size: int = 96
    model_input_size: int = 88
    grayscale: bool = True
    normalize_mean: float = 0.421
    normalize_std: float = 0.165
    smoothing_window: int = 12
    live_smoothing_window: int = 6

    def __post_init__(self) -> None:
        # YAML gives lists; keep the dataclass hashable/immutable.
        object.__setattr__(self, "stable_points", tuple(self.stable_points))


@dataclass(frozen=True)
class ModelConfig:
    name: str = "auto_avsr_vsr"
    device: Literal["auto", "mps", "cpu"] = "auto"
    checkpoint: str = "models/vsr/model.pth"
    model_config: str = "models/vsr/model.json"
    tokenizer: str = "models/vsr/unigram5000.model"
    tokenizer_units: str = "models/vsr/unigram5000_units.txt"
    language_model: str | None = None
    language_model_config: str | None = None   # default: language_model with .json suffix
    # Fine-tuned weights (Phase 4): a run directory holding best.pth (or a .pth file). Used
    # instead of `checkpoint` when it exists; null always uses the pretrained checkpoint.
    finetuned: str | None = "models/finetuned/current"


@dataclass(frozen=True)
class DecodeConfig:
    beam_size: int = 40
    ctc_weight: float = 0.1
    lm_weight: float = 0.3
    length_bonus: float = 0.0
    top_k: int = 3


@dataclass(frozen=True)
class DataConfig:
    personal_dir: str = "data/personal"
    manifest: str = "data/personal/manifest.jsonl"


@dataclass(frozen=True)
class RecordConfig:
    prompts: str | None = None         # optional extra prompt file, one phrase per line
    count: int = 100                   # takes to keep per session
    takes_per_phrase: int = 5          # stop prompting a phrase once it has this many takes
    min_frames: int = 13               # ~0.5 s at 25 fps
    max_frames: int = 250              # 10 s
    min_detection_rate: float = 0.9    # fraction of frames with a detected face
    save_video: bool = True            # also keep the full-frame video.mp4 per clip
    baseline: bool = True              # transcribe each take with the current model
    val_fraction: float = 0.1          # deterministic per-clip train/val split


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "models/finetuned"   # runs go to <output_dir>/<timestamp>/
    trainable_blocks: int = 4          # top Conformer blocks trained (frontend/embed frozen)
    epochs: int = 10
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1       # linear warmup over this share of steps, then cosine
    max_batch_frames: int = 800        # padded frames per batch (B * T_max)
    grad_clip: float = 5.0
    ctc_weight: float = 0.1            # loss = w * CTC + (1 - w) * attention CE
    label_smoothing: float = 0.1
    patience: int = 3                  # stop after this many epochs without a better val WER
    min_clips: int = 100               # refuse a real run below this unless --force
    eval_beam_size: int | None = None  # null = decode.beam_size
    random_crop: bool = True           # augmentation: random 88 crop from the 96 ROI
    hflip_prob: float = 0.5            # augmentation: horizontal flip probability
    time_mask_window: int = 10         # augmentation: mask up to this many frames ...
    time_mask_stride: int = 25         # ... once per this many frames
    seed: int = 0


@dataclass(frozen=True)
class AppConfig:
    hotkey: str = "right_option"       # see mouthpiece.app.hotkey.HOTKEYS
    mode: Literal["command", "dictation"] = "command"
    min_hold_s: float = 0.3            # shorter presses are ignored (a tap of the key)
    max_hold_s: float = 15.0           # a longer hold stops and transcribes automatically
    picker_timeout_s: float = 2.0      # top-3 picker auto-accepts #1 after this; 0 = wait
    dictation_picker: bool = False     # false: dictation types #1 at once, no picker
    commands: str | None = None        # extra command phrases, one per line (`#` comments)


@dataclass(frozen=True)
class TypingConfig:
    enabled: bool = True               # false: `mouthpiece app` only prints commits
    method: Literal["auto", "keys", "paste"] = "auto"
    paste_over_chars: int = 200        # auto: paste text longer than this instead of typing
    paste_apps: tuple[str, ...] = ()   # auto: bundle ids that get paste (ignore unicode events)
    clipboard_restore_s: float = 0.5   # old clipboard is put back this long after ⌘V
    key_delay_ms: float = 2.0          # pause after each posted event
    history: int = 20                  # typed chunks "scratch that" can remove, newest first

    def __post_init__(self) -> None:
        object.__setattr__(self, "paste_apps", tuple(self.paste_apps or ()))


@dataclass(frozen=True)
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    app: AppConfig = field(default_factory=AppConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    root: Path = REPO_ROOT

    def resolve(self, path: str | Path) -> Path:
        """Resolve a config path relative to the repo root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.root / p


class ConfigError(ValueError):
    """Raised when the config file is malformed."""


T = TypeVar("T")


def _build(cls: type[T], section: str, raw: Any) -> T:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{section}] unknown keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path`, $MOUTHPIECE_CONFIG, or configs/mouthpiece.yaml."""
    cfg_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top level of config must be a mapping")
    sections: dict[str, type[Any]] = {
        "capture": CaptureConfig,
        "preprocess": PreprocessConfig,
        "model": ModelConfig,
        "decode": DecodeConfig,
        "data": DataConfig,
        "record": RecordConfig,
        "train": TrainConfig,
        "app": AppConfig,
        "typing": TypingConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
    built = {name: _build(cls, name, raw.get(name)) for name, cls in sections.items()}
    cfg = Config(**built)
    if cfg.model.device not in ("auto", "mps", "cpu"):
        raise ConfigError(f"model.device must be auto|mps|cpu, got {cfg.model.device!r}")
    if cfg.preprocess.model_input_size > cfg.preprocess.roi_size:
        raise ConfigError("preprocess.model_input_size must be <= roi_size")
    r = cfg.record
    if not 0 < r.min_frames <= r.max_frames:
        raise ConfigError("record: need 0 < min_frames <= max_frames")
    if not 0.0 <= r.val_fraction < 1.0 or not 0.0 <= r.min_detection_rate <= 1.0:
        raise ConfigError("record: val_fraction must be in [0, 1), min_detection_rate in [0, 1]")
    if r.count < 1 or r.takes_per_phrase < 1:
        raise ConfigError("record: count and takes_per_phrase must be >= 1")
    t = cfg.train
    if t.trainable_blocks < 0 or t.epochs < 1 or t.max_batch_frames < 1 or t.min_clips < 0:
        raise ConfigError("train: need trainable_blocks >= 0, epochs >= 1, max_batch_frames >= 1")
    if not 0.0 <= t.ctc_weight <= 1.0 or not 0.0 <= t.hflip_prob <= 1.0:
        raise ConfigError("train: ctc_weight and hflip_prob must be in [0, 1]")
    if not 0.0 <= t.warmup_fraction < 1.0 or not 0.0 <= t.label_smoothing < 1.0:
        raise ConfigError("train: warmup_fraction and label_smoothing must be in [0, 1)")
    if t.lr <= 0 or t.patience < 1 or t.time_mask_stride < 1 or t.time_mask_window < 0:
        raise ConfigError("train: need lr > 0, patience >= 1, time_mask_stride >= 1")
    a = cfg.app
    from mouthpiece.app.hotkey import HOTKEYS

    if a.hotkey not in HOTKEYS:
        raise ConfigError(f"app.hotkey must be one of {sorted(HOTKEYS)}, got {a.hotkey!r}")
    if a.mode not in ("command", "dictation"):
        raise ConfigError(f"app.mode must be command|dictation, got {a.mode!r}")
    if a.min_hold_s < 0 or a.max_hold_s < 0 or a.picker_timeout_s < 0:
        raise ConfigError("app: min_hold_s, max_hold_s and picker_timeout_s must be >= 0")
    ty = cfg.typing
    if ty.method not in ("auto", "keys", "paste"):
        raise ConfigError(f"typing.method must be auto|keys|paste, got {ty.method!r}")
    if (ty.paste_over_chars < 0 or ty.clipboard_restore_s < 0 or ty.key_delay_ms < 0
            or ty.history < 1):
        raise ConfigError("typing: need paste_over_chars, clipboard_restore_s, key_delay_ms "
                          ">= 0 and history >= 1")
    return cfg
