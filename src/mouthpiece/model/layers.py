"""Transformer / Conformer building blocks used by the Auto-AVSR model and its LM.

Ported (and trimmed to inference) from ESPnet, as shipped in
mpc001/Visual_Speech_Recognition_for_Multiple_Languages (espnet/nets/pytorch_backend/
transformer/*). Original code: Copyright 2019 Shigeki Karita, 2020 Johns Hopkins
University (Shinji Watanabe), Northwestern Polytechnical University (Pengcheng Guo).
Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

Module and parameter names are kept identical so the original state dicts load as-is
(including the upstream typo `pointwise_cov1`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class LayerNorm(nn.LayerNorm):
    """ESPnet's LayerNorm: torch LayerNorm with eps=1e-12."""

    def __init__(self, nout: int) -> None:
        super().__init__(nout, eps=1e-12)


def subsequent_mask(size: int, device: torch.device | None = None) -> Tensor:
    """(size, size) bool lower-triangular mask: position i may attend to j <= i."""
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.w_2(torch.relu(self.w_1(x)))
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int) -> None:
        super().__init__()
        if n_feat % n_head:
            raise ValueError("n_feat must be divisible by n_head")
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def _heads(self, x: Tensor) -> Tensor:
        """(B, T, F) -> (B, H, T, d_k)."""
        return x.view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

    def project_q(self, query: Tensor) -> Tensor:
        return self._heads(self.linear_q(query))

    def project_kv(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        return self._heads(self.linear_k(key)), self._heads(self.linear_v(value))

    def attend(self, v: Tensor, scores: Tensor, mask: Tensor | None) -> Tensor:
        """scores (B, H, T1, T2); mask (B, 1|T1, T2) bool, True = keep."""
        if mask is not None:
            drop = ~mask.unsqueeze(1)
            scores = scores.masked_fill(drop, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1).masked_fill(drop, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v)  # (B, H, T1, d_k)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)
        out: Tensor = self.linear_out(x)
        return out

    def forward_kv(self, query: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        """Attention with already-projected keys/values (B or 1, H, T2, d_k)."""
        if k.size(0) == 1 and query.size(0) > 1 and mask is None:
            # Keys/values shared by the whole batch (e.g. one utterance, many beam
            # hypotheses): fold the batch into the query length instead of broadcasting,
            # which is much faster on MPS.
            b, t1 = query.size(0), query.size(1)
            out = self.forward_kv(query.reshape(1, b * t1, -1), k, v, None)
            return out.view(b, t1, -1)
        q = self.project_q(query)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.attend(v, scores, mask)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
        k, v = self.project_kv(key, value)
        return self.forward_kv(query, k, v, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Self-attention with relative positional encoding (ESPnet "latest" rel_pos_type)."""

    def __init__(self, n_head: int, n_feat: int, zero_triu: bool = False) -> None:
        super().__init__(n_head, n_feat)
        self.zero_triu = zero_triu
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: Tensor) -> Tensor:
        """(B, H, T1, 2*T1-1) -> (B, H, T1, T1)."""
        b, h, t1, n = x.shape
        zero_pad = torch.zeros((b, h, t1, 1), device=x.device, dtype=x.dtype)
        padded = torch.cat([zero_pad, x], dim=-1).view(b, h, n + 1, t1)
        out = padded[:, :, 1:].reshape(b, h, t1, n)[:, :, :, : n // 2 + 1]
        if self.zero_triu:
            ones = torch.ones((out.size(2), out.size(3)), device=x.device)
            out = out * torch.tril(ones, out.size(3) - out.size(2))[None, None]
        return out

    def forward_rel(
        self, query: Tensor, key: Tensor, value: Tensor, pos_emb: Tensor, mask: Tensor | None
    ) -> Tensor:
        q = self.project_q(query).transpose(1, 2)  # (B, T1, H, d_k)
        k, v = self.project_kv(key, value)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        matrix_bd = self.rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.attend(v, scores, mask)


def _sinusoid(positions: Tensor, d_model: int) -> Tensor:
    pe = torch.zeros(positions.numel(), d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    arg = positions.float().unsqueeze(1) * div_term
    pe[:, 0::2] = torch.sin(arg)
    pe[:, 1::2] = torch.cos(arg)
    return pe


class PositionalEncoding(nn.Module):
    """Absolute sinusoidal encoding: x * sqrt(d) + PE."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.pe = _sinusoid(torch.arange(max_len), d_model).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        if self.pe.size(1) < x.size(1):
            self.pe = _sinusoid(torch.arange(x.size(1)), self.d_model).unsqueeze(0)
        if self.pe.device != x.device or self.pe.dtype != x.dtype:
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x * self.xscale + self.pe[:, : x.size(1)]


class RelPositionalEncoding(nn.Module):
    """Relative encoding (ESPnet PR #2816): returns (x * sqrt(d), pos_emb (1, 2T-1, d))."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.pe = self._build(max_len)

    def _build(self, length: int) -> Tensor:
        pos = torch.arange(length)
        positive = torch.flip(_sinusoid(pos, self.d_model), [0])
        negative = _sinusoid(-pos, self.d_model)[1:]
        return torch.cat([positive, negative], dim=0).unsqueeze(0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = x.size(1)
        if self.pe.size(1) < 2 * t - 1:
            self.pe = self._build(t)
        if self.pe.device != x.device or self.pe.dtype != x.dtype:
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        center = self.pe.size(1) // 2
        return x * self.xscale, self.pe[:, center - t + 1 : center + t]


class ConvolutionModule(nn.Module):
    """Conformer convolution module: pointwise -> GLU -> depthwise -> BN -> swish -> pointwise."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if (kernel_size - 1) % 2:
            raise ValueError("kernel_size must be odd")
        self.pointwise_cov1 = nn.Conv1d(channels, 2 * channels, 1)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_cov2 = nn.Conv1d(channels, channels, 1)
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = nn.functional.glu(self.pointwise_cov1(x), dim=1)
        x = self.activation(self.norm(self.depthwise_conv(x)))
        out: Tensor = self.pointwise_cov2(x)
        return out.transpose(1, 2)
