"""Batched joint CTC/attention beam search with an optional language model.

Follows ESPnet's `BatchBeamSearch` as configured by the Auto-AVSR inference pipeline
(Apache License 2.0):

    score = (1 - ctc_weight) * decoder + ctc_weight * ctc + lm_weight * lm
            + length_bonus * num_tokens

* Each step, the attention decoder (and LM) score every token for every running hypothesis.
* CTC prefix scores, the expensive part, are computed only for the top `1.5 * beam` tokens
  of each hypothesis by decoder score ("pre-beam"); everything else gets log(0).
* The best `beam` (hypothesis, token) pairs survive; those ending in <eos> are finished.
* With no explicit max length, decoding runs up to one token per encoder frame and stops
  early via ESPnet's end detection (no better hypothesis ended in the last 3 lengths).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mouthpiece.decode.ctc_prefix import CTCPrefixScorer, CTCState

# A step scorer maps prefixes (N, L) + its cache to next-token log-probs (N, V) + new cache.
Cache = list[Tensor]
StepFn = Callable[[Tensor, Cache | None], tuple[Tensor, Cache]]


@dataclass(frozen=True)
class Hypothesis:
    tokens: tuple[int, ...]              # without sos/eos
    score: float                          # weighted total
    scores: dict[str, float] = field(default_factory=dict)  # per-scorer cumulative log-probs

    @property
    def length(self) -> int:
        """ESPnet yseq length: sos + tokens + eos."""
        return len(self.tokens) + 2


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_size: int = 40
    ctc_weight: float = 0.1
    lm_weight: float = 0.3
    length_bonus: float = 0.0
    pre_beam_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if not 0.0 <= self.ctc_weight <= 1.0:
            raise ValueError("ctc_weight must be in [0, 1]")


def end_detect(ended: list[Hypothesis], i: int, m: int = 3, d_end: float = -10.0) -> bool:
    """ESPnet end detection (Watanabe et al. 2017, eq. 50)."""
    if not ended:
        return False
    best = max(h.score for h in ended)
    count = 0
    for k in range(m):
        same = [h.score for h in ended if h.length == i - k]
        if same and max(same) - best < d_end:
            count += 1
    return count == m


def _select(cache: Cache | None, idx: Tensor) -> Cache | None:
    return None if cache is None else [c.index_select(0, idx) for c in cache]


class BeamSearch:
    def __init__(
        self,
        decoder: StepFn,
        vocab_size: int,
        sos: int,
        eos: int,
        cfg: BeamSearchConfig,
        ctc_logp: Tensor | None = None,
        lm: StepFn | None = None,
        blank: int = 0,
    ) -> None:
        """decoder / lm: step functions bound to this utterance (decoder holds the encoder
        output). ctc_logp: (T, V) CTC log-posteriors; scored on the device it lives on."""
        self.decoder, self.lm = decoder, lm
        self.vocab_size, self.sos, self.eos = vocab_size, sos, eos
        self.cfg = cfg
        self.ctc: CTCPrefixScorer | None = None
        if ctc_logp is not None and cfg.ctc_weight > 0:
            self.ctc = CTCPrefixScorer(ctc_logp, blank, eos)
        self.dec_weight = 1.0 - cfg.ctc_weight if self.ctc is not None else 1.0
        self.pre_beam = int(cfg.pre_beam_ratio * cfg.beam_size)

    def __call__(self, device: torch.device, maxlen: int) -> list[Hypothesis]:
        """Return finished hypotheses sorted best-first. maxlen: max tokens incl. <eos>."""
        cfg, V = self.cfg, self.vocab_size
        ys = torch.full((1, 1), self.sos, dtype=torch.long, device=device)
        total = torch.zeros(1, device=device)
        parts: dict[str, Tensor] = {"decoder": torch.zeros(1, device=device)}
        if self.ctc is not None:
            parts["ctc"] = torch.zeros(1, device=device)
        if self.lm is not None:
            parts["lm"] = torch.zeros(1, device=device)
        dec_cache: Cache | None = None
        lm_cache: Cache | None = None
        ctc_state: CTCState | None = self.ctc.init_state() if self.ctc is not None else None
        ended: list[Hypothesis] = []

        for i in range(maxlen):
            dec_logp, dec_cache_all = self.decoder(ys, dec_cache)
            step = {"decoder": dec_logp}
            weighted = self.dec_weight * dec_logp
            if self.lm is not None:
                lm_logp, lm_cache_all = self.lm(ys, lm_cache)
                step["lm"] = lm_logp
                weighted = weighted + cfg.lm_weight * lm_logp
            if cfg.length_bonus:
                weighted = weighted + cfg.length_bonus
            ctc_out = None
            if self.ctc is not None and ctc_state is not None:
                ctc_dev = self.ctc.x.device
                part_ids = None
                if self.pre_beam < V:
                    part_ids = torch.topk(dec_logp, self.pre_beam, dim=-1).indices.to(ctc_dev)
                ctc_out = self.ctc.score(ys[:, -1].to(ctc_dev), i, ctc_state, part_ids)
                step["ctc"] = ctc_out.scores.to(device)
                weighted = weighted + cfg.ctc_weight * step["ctc"]
            weighted = weighted + total.unsqueeze(1)

            k = min(cfg.beam_size, weighted.numel())
            top = weighted.view(-1).topk(k)
            prev = torch.div(top.indices, V, rounding_mode="trunc")
            tok = top.indices % V
            ys = torch.cat([ys[prev], tok.unsqueeze(1)], dim=1)
            total = top.values
            parts = {name: parts[name][prev] + step[name][prev, tok] for name in parts}
            dec_cache = _select(dec_cache_all, prev)
            if self.lm is not None:
                lm_cache = _select(lm_cache_all, prev)
            if ctc_out is not None:
                ctc_dev = ctc_out.r.device
                ctc_state = CTCPrefixScorer.select(ctc_out, prev.to(ctc_dev), tok.to(ctc_dev))

            if i == maxlen - 1:  # force-finish everything still running
                is_eos = torch.ones(k, dtype=torch.bool)
            else:
                is_eos = (tok == self.eos).cpu()
            if bool(is_eos.any()):
                ys_cpu, total_cpu = ys.cpu(), total.cpu()
                parts_cpu = {n_: v.cpu() for n_, v in parts.items()}
                for b in torch.nonzero(is_eos).view(-1).tolist():
                    seq = ys_cpu[b, 1:].tolist()
                    if seq and seq[-1] == self.eos:
                        seq = seq[:-1]
                    ended.append(Hypothesis(
                        tuple(int(t) for t in seq), float(total_cpu[b]),
                        {n_: float(v[b]) for n_, v in parts_cpu.items()},
                    ))
                keep = torch.nonzero(~is_eos).view(-1).to(device)
                if keep.numel() == 0:
                    break
                ys, total = ys[keep], total[keep]
                parts = {n_: v[keep] for n_, v in parts.items()}
                dec_cache = _select(dec_cache, keep)
                lm_cache = _select(lm_cache, keep)
                if ctc_state is not None:
                    kc = keep.to(ctc_state.r.device)
                    ctc_state = CTCState(ctc_state.r[:, :, kc], ctc_state.s[kc])
            if end_detect(ended, i):
                break
        return sorted(ended, key=lambda h: h.score, reverse=True)

