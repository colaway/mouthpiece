"""Torch device selection (MPS on Apple Silicon, CPU otherwise)."""

from __future__ import annotations

import torch


def resolve_device(preferred: str = "auto") -> torch.device:
    """'auto' -> MPS when available, else CPU. 'mps' raises if MPS is unavailable.

    Ops MPS lacks fall back to CPU because the package sets PYTORCH_ENABLE_MPS_FALLBACK=1
    before torch is imported (see mouthpiece/__init__.py).
    """
    if preferred == "cpu":
        return torch.device("cpu")
    mps = torch.backends.mps.is_available()
    if preferred == "mps" and not mps:
        raise RuntimeError("MPS requested but not available on this machine")
    if preferred not in ("auto", "mps"):
        raise ValueError(f"device must be auto|mps|cpu, got {preferred!r}")
    return torch.device("mps" if mps else "cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
