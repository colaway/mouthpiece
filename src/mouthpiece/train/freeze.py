"""Which parameters fine-tuning updates, and train/eval mode handling for the partial model.

Frozen: the video frontend (3D conv + ResNet-18), `encoder.embed` and the lower Conformer
blocks. Trained: the top `trainable_blocks` Conformer blocks, `encoder.after_norm`, the whole
decoder and the CTC head. Every BatchNorm stays in eval mode (frozen running statistics),
also inside trained blocks: personal batches are far too small for stable batch statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from mouthpiece.model.vsr import VSRModel


@dataclass(frozen=True)
class FreezeReport:
    trainable: int
    total: int
    frozen_blocks: int        # lower Conformer blocks that run without gradients
    trainable_blocks: int

    def __str__(self) -> str:
        return (f"{self.trainable / 1e6:.1f}M of {self.total / 1e6:.1f}M parameters trainable "
                f"(top {self.trainable_blocks} Conformer blocks + decoder + CTC)")


def frozen_block_count(model: VSRModel, trainable_blocks: int) -> int:
    n = len(model.encoder.encoders)
    if not 0 <= trainable_blocks <= n:
        raise ValueError(f"train.trainable_blocks must be in [0, {n}], got {trainable_blocks}")
    return n - trainable_blocks


def apply_freeze(model: VSRModel, trainable_blocks: int) -> FreezeReport:
    """Set `requires_grad` per the fine-tuning recipe; return parameter counts."""
    k = frozen_block_count(model, trainable_blocks)
    model.requires_grad_(False)
    enc = model.encoder
    for block in list(enc.encoders)[k:]:
        block.requires_grad_(True)
    enc.after_norm.requires_grad_(True)
    model.decoder.requires_grad_(True)
    model.ctc.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return FreezeReport(trainable, total, k, trainable_blocks)


def set_train_mode(model: VSRModel, frozen_blocks: int) -> None:
    """Train mode for the trained parts only; frozen parts and every BatchNorm stay in eval.

    The port has no dropout, so this changes no math today; it keeps BatchNorm statistics
    frozen and makes any dropout added later behave correctly.
    """
    model.train()
    model.encoder.frontend.eval()
    model.encoder.embed.eval()
    for block in list(model.encoder.encoders)[:frozen_blocks]:
        block.eval()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
