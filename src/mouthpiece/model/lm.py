"""Transformer subword language model (ESPnet `lm/transformer.py`, Apache License 2.0).

The released LRS3 LM: 16 layers, 512-d attention, 8 heads, 2048 FFN units, 128-d token
embedding, no positional encoding (pos_enc: none), vocab 5049 shared with the VSR model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mouthpiece.model.layers import LayerNorm, subsequent_mask
from mouthpiece.model.transformer import Cache, TransformerLayer
from mouthpiece.model.vsr import ModelLoadError, load_state_dict


@dataclass(frozen=True)
class LMConfig:
    n_vocab: int = 5049
    layer: int = 16
    unit: int = 2048
    att_unit: int = 512
    embed_unit: int = 128
    head: int = 8

    @classmethod
    def from_json(cls, path: str | Path) -> LMConfig:
        raw: dict[str, Any] = json.loads(Path(path).read_text())
        if raw.get("model_module") != "transformer":
            raise ModelLoadError(f"unsupported LM type {raw.get('model_module')!r}")
        if raw.get("pos_enc", "none") != "none" or raw.get("tie_weights"):
            raise ModelLoadError("only LMs with pos_enc=none and untied weights are supported")
        return cls(**{k: int(raw[k]) for k in ("n_vocab", "layer", "unit", "att_unit",
                                                "embed_unit", "head")})


class _LMEncoder(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        # input_layer="linear": Linear -> torch LayerNorm (eps 1e-5) -> Dropout -> ReLU
        self.embed = nn.Sequential(
            nn.Linear(cfg.embed_unit, cfg.att_unit),
            nn.LayerNorm(cfg.att_unit),
            nn.Dropout(0.0),
            nn.ReLU(),
        )
        self.encoders = nn.ModuleList(
            TransformerLayer(cfg.att_unit, cfg.head, cfg.unit) for _ in range(cfg.layer)
        )
        self.after_norm = LayerNorm(cfg.att_unit)


class TransformerLM(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.n_vocab, cfg.embed_unit)
        self.encoder = _LMEncoder(cfg)
        self.decoder = nn.Linear(cfg.att_unit, cfg.n_vocab)

    def forward_one_step(self, ys: Tensor, cache: Cache | None = None) -> tuple[Tensor, Cache]:
        """ys (B, L) prefixes -> next-token log-probs (B, V) and new per-layer cache."""
        mask = (ys != 0).unsqueeze(-2) & subsequent_mask(ys.size(-1), ys.device).unsqueeze(0)
        x = self.encoder.embed(self.embed(ys))
        new_cache: Cache = []
        for i, layer in enumerate(self.encoder.encoders):
            x = layer(x, mask, None if cache is None else cache[i])
            new_cache.append(x)
        h = self.encoder.after_norm(x[:, -1])
        return torch.log_softmax(self.decoder(h), dim=-1), new_cache


def load_lm(
    checkpoint: str | Path, config: str | Path, device: torch.device | str = "cpu"
) -> TransformerLM:
    lm = TransformerLM(LMConfig.from_json(config))
    lm.load_state_dict(load_state_dict(checkpoint), strict=True)
    return lm.to(device).eval()
