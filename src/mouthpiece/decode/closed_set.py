"""Closed-set decoding for command mode: rank a fixed list of phrases instead of searching.

Each phrase's exact token sequence is scored with the same joint objective the beam search
uses, without the LM (every command is equally likely a priori):

    score = (1 - ctc_weight) * log P_att(y + <eos> | x) + ctc_weight * log P_ctc(y | x)

The attention term is one teacher-forced decoder pass over all phrases at once; the CTC term
is the full CTC forward probability (torch's ctc_loss, on CPU: MPS has no CTC kernel).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mouthpiece.metrics import normalize_text
from mouthpiece.model.transformer import TransformerDecoder


@dataclass(frozen=True)
class PhraseSet:
    texts: tuple[str, ...]
    tokens: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.tokens):
            raise ValueError("texts and tokens must have the same length")
        if not self.texts:
            raise ValueError("a phrase set needs at least one phrase")
        if any(not t for t in self.tokens):
            raise ValueError("every phrase needs at least one token")

    def __len__(self) -> int:
        return len(self.texts)

    @classmethod
    def build(cls, texts: Iterable[str], encode: Callable[[str], list[int]]) -> PhraseSet:
        """Tokenize, dropping empty and duplicate phrases (by normalized text)."""
        seen: set[str] = set()
        out_t: list[str] = []
        out_ids: list[tuple[int, ...]] = []
        for text in texts:
            key = normalize_text(text)
            if not key or key in seen:
                continue
            ids = tuple(encode(text))
            if ids:
                seen.add(key)
                out_t.append(text)
                out_ids.append(ids)
        return cls(tuple(out_t), tuple(out_ids))


def attention_logprobs(
    decoder: TransformerDecoder, enc: Tensor, seqs: Sequence[Sequence[int]], sos: int, eos: int
) -> Tensor:
    """log P(seq + <eos> | enc) for each sequence, teacher-forced in one batch. enc: (T, D)."""
    n = len(seqs)
    lengths = torch.tensor([len(s) + 1 for s in seqs])
    width = int(lengths.max())
    dev = enc.device
    ys_in = torch.full((n, width), eos, dtype=torch.long)
    ys_out = torch.full((n, width), -1, dtype=torch.long)
    ys_in[:, 0] = sos
    for i, s in enumerate(seqs):
        ys_in[i, 1 : len(s) + 1] = torch.tensor(s, dtype=torch.long)
        ys_out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        ys_out[i, len(s)] = eos
    valid = torch.arange(width)[None, :] < lengths[:, None]           # (N, L)
    causal = torch.ones(width, width, dtype=torch.bool).tril()
    mask = (causal[None] & valid[:, None, :]).to(dev)                  # (N, L, L)
    # Project the encoder output once and share it across the batch (expand, no copy).
    kv = [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1))
          for k, v in decoder.memory_kv(enc.unsqueeze(0))]
    x = decoder.embed(ys_in.to(dev))
    for layer, layer_kv in zip(decoder.decoders, kv, strict=True):
        x = layer(x, mask, layer_kv, None)
    logp = torch.log_softmax(decoder.output_layer(decoder.after_norm(x)), dim=-1).float().cpu()
    picked = logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * valid).sum(dim=1)


def ctc_logprobs(ctc_logp: Tensor, seqs: Sequence[Sequence[int]], blank: int = 0) -> Tensor:
    """log P_ctc(seq | x) for each sequence; -inf when it cannot fit. ctc_logp: (T, V)."""
    logp = ctc_logp.float().cpu()
    t, v = logp.shape
    n = len(seqs)
    targets = torch.tensor([tok for s in seqs for tok in s], dtype=torch.long)
    loss = F.ctc_loss(
        logp.unsqueeze(1).expand(t, n, v),
        targets,
        torch.full((n,), t, dtype=torch.long),
        torch.tensor([len(s) for s in seqs], dtype=torch.long),
        blank=blank, reduction="none", zero_infinity=False,
    )
    return -loss


@dataclass(frozen=True)
class RankedPhrase:
    index: int
    text: str
    score: float
    att: float
    ctc: float


def rank(att: Tensor, ctc: Tensor | None, texts: Sequence[str],
         ctc_weight: float) -> list[RankedPhrase]:
    """Combine per-phrase scores and sort best-first."""
    w = ctc_weight if ctc is not None else 0.0
    ctc_vals = ctc if ctc is not None else torch.zeros_like(att)
    total = (1.0 - w) * att + w * ctc_vals if w > 0 else att
    order = torch.argsort(total, descending=True).tolist()
    return [RankedPhrase(i, texts[i], float(total[i]), float(att[i]), float(ctc_vals[i]))
            for i in order]
