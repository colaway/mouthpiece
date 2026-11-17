"""Vectorized CTC prefix scoring for hybrid CTC/attention beam search.

A port of ESPnet's `CTCPrefixScoreTH` (Apache License 2.0; Watanabe et al. 2017, Algorithm 2;
Seki et al. 2019) specialised to one utterance, with the per-hypothesis Python loops replaced
by tensor ops. Scores match the original.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

LOGZERO = -1e10


@dataclass(frozen=True)
class CTCState:
    """Per-hypothesis prefix state: forward variables r (T, 2, N) and prefix score s (N,)."""

    r: Tensor
    s: Tensor


@dataclass(frozen=True)
class CTCScores:
    """Output of one scoring step, needed to build the next states."""

    scores: Tensor       # (N, V) incremental log-prob of each next token (LOGZERO if unscored)
    r: Tensor            # (T, 2, N, S) forward variables of every scored extension
    log_psi: Tensor      # (N, V) full prefix scores
    scoring_ids: Tensor  # (N, S) token ids that were scored


class CTCPrefixScorer:
    def __init__(self, logp: Tensor, blank: int, eos: int) -> None:
        """logp: (T, V) CTC log-posteriors for one utterance."""
        if logp.dim() != 2:
            raise ValueError("logp must be (T, V)")
        self.x = logp
        self.T, self.V = logp.shape
        self.blank, self.eos = blank, eos
        self.x_blank = logp[:, blank]  # (T,)

    def init_state(self) -> CTCState:
        r = torch.full((self.T, 2, 1), LOGZERO, dtype=self.x.dtype, device=self.x.device)
        r[:, 1, 0] = torch.cumsum(self.x_blank, 0)
        return CTCState(r, torch.zeros(1, dtype=self.x.dtype, device=self.x.device))

    def score(
        self, last_ids: Tensor, output_length: int, state: CTCState, scoring_ids: Tensor | None
    ) -> CTCScores:
        """Score extensions of N prefixes.

        last_ids: (N,) last token of each prefix (sos for the empty prefix).
        output_length: number of real tokens in the prefixes (excluding sos).
        scoring_ids: (N, S) candidate tokens per prefix, or None for the full vocabulary.
        """
        n = last_ids.numel()
        if scoring_ids is None:
            scoring_ids = torch.arange(self.V, device=self.x.device).expand(n, self.V)
        snum = scoring_ids.size(1)
        r_prev, s_prev = state.r, state.s
        x_ = self.x[:, scoring_ids]  # (T, N, S)
        xb = self.x_blank.view(-1, 1, 1)

        r = torch.full((self.T, 2, n, snum), LOGZERO, dtype=self.x.dtype, device=self.x.device)
        if output_length == 0:
            r[0, 0] = x_[0]
        r_sum = torch.logsumexp(r_prev, 1)  # (T, N)
        # Repeating the last label needs a blank in between: use r^b only for that label.
        same = (scoring_ids == last_ids.view(-1, 1)).unsqueeze(0)  # (1, N, S)
        log_phi = torch.where(same, r_prev[:, 1].unsqueeze(2), r_sum.unsqueeze(2))  # (T, N, S)

        start = max(output_length, 1)
        for t in range(start, self.T):
            rp = r[t - 1]
            r[t, 0] = torch.logaddexp(rp[0], log_phi[t - 1]) + x_[t]
            r[t, 1] = torch.logaddexp(rp[0], rp[1]) + xb[t]

        log_phi_x = torch.cat((log_phi[:1], log_phi[:-1]), dim=0) + x_
        psi_scored = torch.logsumexp(
            torch.cat((log_phi_x[start:], r[start - 1, 0].unsqueeze(0)), dim=0), dim=0
        )  # (N, S)
        log_psi = torch.full((n, self.V), LOGZERO, dtype=self.x.dtype, device=self.x.device)
        log_psi.scatter_(1, scoring_ids, psi_scored)
        log_psi[:, self.eos] = r_sum[-1]
        log_psi[:, self.blank] = LOGZERO
        return CTCScores(log_psi - s_prev.unsqueeze(1), r, log_psi, scoring_ids)

    @staticmethod
    def select(out: CTCScores, prev: Tensor, tokens: Tensor) -> CTCState:
        """States for the chosen (prefix index, next token) pairs."""
        match = out.scoring_ids[prev] == tokens.unsqueeze(1)  # (K, S)
        pos = match.float().argmax(dim=1)  # 0 if not scored (only happens for eos)
        r = out.r[:, :, prev, pos]  # (T, 2, K)
        s = out.log_psi[prev, tokens]
        return CTCState(r, s)
