"""Torch dataset over the personal recordings (`manifest.jsonl`), for Phase 4 fine-tuning.

Each item is the exact model input of a clip, (T, 88, 88) float32 center-cropped and
normalized like inference, plus its target token ids. `collate` pads a batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mouthpiece.config import Config
from mouthpiece.preprocess.mouth_roi import to_model_input
from mouthpiece.record.manifest import ManifestEntry, Split, read_manifest

IGNORE_ID = -1   # ESPnet's padding value for token targets

# Optional augmentation on the uint8 (T, 96, 96) crops before the 88x88 center crop.
CropTransform = Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]]


@dataclass(frozen=True)
class Sample:
    video: Tensor        # (T, 88, 88) float32
    tokens: Tensor       # (L,) int64
    id: str
    text: str


@dataclass(frozen=True)
class Batch:
    videos: Tensor         # (B, T_max, 88, 88) float32, zero-padded
    video_lengths: Tensor  # (B,) int64
    tokens: Tensor         # (B, L_max) int64, padded with IGNORE_ID
    token_lengths: Tensor  # (B,) int64
    ids: list[str]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    def to(self, device: torch.device | str) -> Batch:
        return Batch(self.videos.to(device), self.video_lengths.to(device),
                     self.tokens.to(device), self.token_lengths.to(device), self.ids, self.texts)


class PersonalDataset(Dataset[Sample]):
    def __init__(
        self,
        root: Path,
        entries: Sequence[ManifestEntry],
        input_size: int = 88,
        mean: float = 0.421,
        std: float = 0.165,
        transform: CropTransform | None = None,
    ) -> None:
        self.root, self.entries = root, list(entries)
        self.input_size, self.mean, self.std = input_size, mean, std
        self.transform = transform

    @classmethod
    def from_config(
        cls, cfg: Config, split: Split | None = None, transform: CropTransform | None = None
    ) -> PersonalDataset:
        """All clips in the manifest, or only the `train` / `val` split."""
        entries = read_manifest(cfg.resolve(cfg.data.manifest))
        if split is not None:
            entries = [e for e in entries if e.split == split]
        p = cfg.preprocess
        return cls(cfg.resolve(cfg.data.personal_dir), entries, p.model_input_size,
                   p.normalize_mean, p.normalize_std, transform)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Sample:
        e = self.entries[i]
        mouths: npt.NDArray[np.uint8] = np.load(self.root / e.path / "mouth.npy",
                                                allow_pickle=False)
        if self.transform is not None:
            mouths = self.transform(mouths)
        x = to_model_input(mouths, self.input_size, self.mean, self.std)
        return Sample(torch.from_numpy(np.ascontiguousarray(x)),
                      torch.tensor(e.tokens, dtype=torch.long), e.id, e.text)

    @property
    def total_frames(self) -> int:
        return sum(e.frames for e in self.entries)


def collate(samples: Sequence[Sample]) -> Batch:
    if not samples:
        raise ValueError("empty batch")
    b = len(samples)
    t_max = max(s.video.shape[0] for s in samples)
    l_max = max(s.tokens.shape[0] for s in samples)
    h, w = samples[0].video.shape[1:]
    videos = torch.zeros(b, t_max, h, w, dtype=torch.float32)
    tokens = torch.full((b, l_max), IGNORE_ID, dtype=torch.long)
    for i, s in enumerate(samples):
        videos[i, : s.video.shape[0]] = s.video
        tokens[i, : s.tokens.shape[0]] = s.tokens
    return Batch(
        videos,
        torch.tensor([s.video.shape[0] for s in samples], dtype=torch.long),
        tokens,
        torch.tensor([s.tokens.shape[0] for s in samples], dtype=torch.long),
        [s.id for s in samples],
        [s.text for s in samples],
    )
