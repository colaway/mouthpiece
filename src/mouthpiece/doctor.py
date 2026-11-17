"""Environment health checks: `mouthpiece doctor`."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from mouthpiece.config import Config


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str = ""


# --- individual checks -----------------------------------------------------


def check_platform() -> CheckResult:
    machine = platform.machine()
    system = platform.system()
    if system == "Darwin" and machine == "arm64":
        return CheckResult("platform", Status.OK, f"macOS {platform.mac_ver()[0]} on {machine}")
    return CheckResult(
        "platform", Status.WARN, f"{system} {machine}", "Mouthpiece targets Apple Silicon Macs."
    )


def check_python() -> CheckResult:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        return CheckResult("python", Status.OK, detail)
    return CheckResult("python", Status.WARN, detail, "Use the uv env: `uv run mouthpiece doctor`.")


def check_torch_device(preferred: str = "auto") -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("torch", Status.FAIL, "PyTorch not installed", "Run `uv sync`.")

    mps_ok = torch.backends.mps.is_available()
    if mps_ok:
        try:
            x = torch.ones(4, device="mps")
            mps_ok = float((x * 2).sum().cpu()) == 8.0
        except Exception as exc:  # pragma: no cover - hardware dependent
            return CheckResult(
                "torch/mps", Status.WARN, f"MPS reported available but failed: {exc}",
                "Falling back to CPU.",
            )
    version = torch.__version__
    if preferred == "cpu":
        return CheckResult("torch/mps", Status.OK, f"torch {version}; device forced to CPU")
    if mps_ok:
        return CheckResult("torch/mps", Status.OK, f"torch {version}; MPS available and working")
    status = Status.FAIL if preferred == "mps" else Status.WARN
    return CheckResult(
        "torch/mps", status, f"torch {version}; MPS unavailable, will use CPU",
        "Requires macOS 12.3+ on Apple Silicon and an arm64 Python.",
    )


# AVAuthorizationStatus values from AVFoundation.
_AV_STATUS = {0: "not determined", 1: "restricted", 2: "denied", 3: "authorized"}


def camera_authorization_status() -> int | None:
    """Return AVAuthorizationStatus for video, or None if AVFoundation is unavailable."""
    try:
        import AVFoundation
    except ImportError:
        return None
    return int(AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeVideo
    ))


def check_camera_permission(
    status_fn: Callable[[], int | None] = camera_authorization_status,
) -> CheckResult:
    status = status_fn()
    fix = (
        "System Settings → Privacy & Security → Camera → enable the app running Mouthpiece "
        "(Terminal, iTerm, VS Code, …), then restart that app."
    )
    if status is None:
        return CheckResult("camera permission", Status.WARN, "AVFoundation not available", fix)
    label = _AV_STATUS.get(status, f"unknown ({status})")
    if status == 3:
        return CheckResult("camera permission", Status.OK, label)
    if status == 0:
        return CheckResult(
            "camera permission", Status.WARN, label,
            "Run `uv run mouthpiece doctor --probe-camera` to trigger the macOS prompt.",
        )
    return CheckResult("camera permission", Status.FAIL, label, fix)


_PERM_FIX = (
    "System Settings → Privacy & Security → {pane} → enable the app running Mouthpiece "
    "(Terminal, iTerm, VS Code, …), then restart that app."
)


def check_input_monitoring(fn: Callable[[], bool | None] | None = None) -> CheckResult:
    """The menu-bar app's global hotkey needs to listen to key events."""
    from mouthpiece.app.permissions import input_monitoring_granted

    granted = (fn or input_monitoring_granted)()
    name = "input monitoring"
    if granted is None:
        return CheckResult(name, Status.WARN, "unavailable (not macOS / no Quartz)")
    if granted:
        return CheckResult(name, Status.OK, "granted (global hotkey for `mouthpiece app`)")
    return CheckResult(name, Status.WARN, "not granted — `mouthpiece app` hotkey won't work",
                       _PERM_FIX.format(pane="Input Monitoring")
                       + " The app's menu has a 'Grant Input Monitoring…' item that prompts.")


def check_accessibility(fn: Callable[[], bool | None] | None = None) -> CheckResult:
    """Typing into other apps (posting key events) and the picker's 1/2/3/Return/Esc keys."""
    from mouthpiece.app.permissions import accessibility_granted

    granted = (fn or accessibility_granted)()
    name = "accessibility"
    if granted is None:
        return CheckResult(name, Status.WARN, "unavailable (not macOS)")
    if granted:
        return CheckResult(name, Status.OK, "granted (typing output, picker keys 1/2/3, ↵, esc)")
    return CheckResult(name, Status.WARN,
                       "not granted — `mouthpiece app` can't type; picker by click/timeout only",
                       _PERM_FIX.format(pane="Accessibility")
                       + " The app's menu has a 'Grant Accessibility…' item that prompts.")


def check_camera_open(index: int, fps: int) -> CheckResult:
    """Open the camera and grab one frame. May trigger the macOS permission prompt."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            return CheckResult(
                "camera capture", Status.FAIL, f"could not open camera {index}",
                "Check camera permission and that no other app holds the camera.",
            )
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        for _ in range(5):  # first frames can be black/empty while the sensor warms up
            if ok and frame is not None:
                break
            ok, frame = cap.read()
        if not ok or frame is None:
            return CheckResult("camera capture", Status.FAIL, "camera opened but returned no frame")
        h, w = frame.shape[:2]
        reported = cap.get(cv2.CAP_PROP_FPS)
        detail = f"{w}x{h}, driver reports {reported:.0f} fps"
        return CheckResult("camera capture", Status.OK, detail)
    finally:
        cap.release()


def check_model_files(cfg: Config) -> list[CheckResult]:
    results: list[CheckResult] = []
    required = {
        "model checkpoint": cfg.model.checkpoint,
        "model config": cfg.model.model_config,
        "tokenizer units": cfg.model.tokenizer_units,
        "tokenizer": cfg.model.tokenizer,
    }
    for label, rel in required.items():
        path = cfg.resolve(rel)
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            results.append(CheckResult(label, Status.OK, f"{rel} ({size_mb:.1f} MB)"))
        else:
            results.append(CheckResult(
                label, Status.WARN, f"missing: {rel}", "Run `./scripts/fetch_models.sh`."
            ))
    units = cfg.resolve(cfg.model.tokenizer_units)
    if units.exists():
        results.append(check_tokenizer(units))
    for label, rel in {
        "face landmarker": cfg.preprocess.face_landmarker_model,
        "mean face": cfg.preprocess.mean_face,
    }.items():
        if cfg.resolve(rel).exists():
            results.append(CheckResult(label, Status.OK, rel))
        else:
            results.append(CheckResult(
                label, Status.FAIL, f"missing: {rel}", "Run `./scripts/fetch_models.sh`."
            ))
    results.append(check_active_checkpoint(cfg))
    if cfg.model.language_model:
        path = cfg.resolve(cfg.model.language_model)
        if path.exists():
            results.append(CheckResult("language model", Status.OK, cfg.model.language_model))
        else:
            results.append(CheckResult(
                "language model", Status.WARN, f"missing (optional): {cfg.model.language_model}"
            ))
    return results


def check_active_checkpoint(cfg: Config) -> CheckResult:
    """Which VSR weights inference loads: an activated fine-tuned run or the pretrained model."""
    from mouthpiece.model.checkpoint import resolve_checkpoint

    active = resolve_checkpoint(cfg)
    if not active.finetuned and not active.path.exists():
        return CheckResult("active weights", Status.WARN, "no checkpoint available",
                           "Run `./scripts/fetch_models.sh`.")
    return CheckResult("active weights", Status.OK, active.describe(cfg))


def check_tokenizer(units_path: Path, expected_vocab: int = 5049) -> CheckResult:
    """The units file must give the vocabulary size of the VSR model's output layer."""
    from mouthpiece.model.tokenizer import Tokenizer

    try:
        vocab = Tokenizer.from_files(units_path).vocab_size
    except (ValueError, OSError) as exc:
        return CheckResult("tokenizer vocab", Status.FAIL, f"unreadable: {exc}",
                           "Re-run `./scripts/fetch_models.sh`.")
    if vocab != expected_vocab:
        return CheckResult("tokenizer vocab", Status.FAIL,
                           f"{vocab} tokens, model expects {expected_vocab}",
                           "Re-run `./scripts/fetch_models.sh`.")
    return CheckResult("tokenizer vocab", Status.OK, f"{vocab} tokens (blank + units + eos)")


def check_personal_data(cfg: Config) -> CheckResult:
    """Informational: how much personal training data has been recorded (Phase 3)."""
    from mouthpiece.record.manifest import ManifestError, read_manifest

    path = cfg.resolve(cfg.data.manifest)
    try:
        entries = read_manifest(path)
    except ManifestError as exc:
        return CheckResult("personal data", Status.WARN, f"manifest unreadable: {exc}",
                           "Run `mouthpiece dataset check`.")
    if not entries:
        return CheckResult("personal data", Status.OK, "no clips yet (`mouthpiece record`)")
    minutes = sum(e.duration_s for e in entries) / 60
    return CheckResult("personal data", Status.OK,
                       f"{len(entries)} clips, {minutes:.1f} min in {cfg.data.personal_dir}")


# --- runner ----------------------------------------------------------------


def run_checks(cfg: Config, probe_camera: bool = False) -> list[CheckResult]:
    results = [
        check_platform(),
        check_python(),
        check_torch_device(cfg.model.device),
        check_camera_permission(),
        check_input_monitoring(),
        check_accessibility(),
    ]
    if probe_camera:
        results.append(check_camera_open(cfg.capture.camera_index, cfg.capture.fps))
    results.extend(check_model_files(cfg))
    results.append(check_personal_data(cfg))
    return results


_ICON = {Status.OK: "✅", Status.WARN: "⚠️ ", Status.FAIL: "❌"}


def format_results(results: Iterable[CheckResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"{_ICON[r.status]} {r.name:<18} {r.detail}")
        if r.fix and r.status is not Status.OK:
            lines.append(f"   ↳ {r.fix}")
    return "\n".join(lines)


def exit_code(results: Iterable[CheckResult]) -> int:
    return 1 if any(r.status is Status.FAIL for r in results) else 0
