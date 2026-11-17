"""Train-time augmentation of mouth crops (Auto-AVSR recipe), on uint8 (T, 96, 96) crops.

Random 88x88 crop instead of the center crop, horizontal flip, and adaptive time masking
(up to `window` frames blacked out once per `stride` frames). The output is (T, 88, 88)
uint8; the dataset's center crop of an 88x88 input is then the identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mouthpiece.config import TrainConfig


@dataclass
class TrainAugment:
    crop: int = 88
    random_crop: bool = True
    hflip_prob: float = 0.5
    time_mask_window: int = 10
    time_mask_stride: int = 25
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @classmethod
    def from_config(cls, t: TrainConfig, crop: int, seed: int | None = None) -> TrainAugment:
        return cls(crop, t.random_crop, t.hflip_prob, t.time_mask_window, t.time_mask_stride,
                   np.random.default_rng(seed))

    def __call__(self, mouths: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        t, h, w = mouths.shape
        c = self.crop
        if c > h or c > w:
            raise ValueError(f"crop {c} larger than input {h}x{w}")
        if self.random_crop:
            y0 = int(self.rng.integers(0, h - c + 1))
            x0 = int(self.rng.integers(0, w - c + 1))
        else:
            y0, x0 = int(round((h - c) / 2.0)), int(round((w - c) / 2.0))
        out = mouths[:, y0 : y0 + c, x0 : x0 + c].copy()
        if self.rng.random() < self.hflip_prob:
            out = out[:, :, ::-1].copy()
        if self.time_mask_window > 0:
            n_masks = int((t + self.time_mask_stride - 0.1) // self.time_mask_stride)
            for _ in range(n_masks):
                width = int(self.rng.integers(0, self.time_mask_window))
                if width == 0 or width >= t:
                    continue
                start = int(self.rng.integers(0, t - width))
                out[start : start + width] = 0
        return out
