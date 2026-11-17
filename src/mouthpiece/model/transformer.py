"""Conformer encoder, transformer decoder and plain transformer stack (for the LM).

Ported (inference only) from ESPnet's pytorch_backend/transformer/{encoder,encoder_layer,
decoder,decoder_layer}.py as shipped with mpc001/Visual_Speech_Recognition_for_Multiple_
Languages. Copyright 2019 Shigeki Karita; Apache License 2.0. Dropout is dropped (inference
only); `normalize_before=True` and `concat_after=False` are the only supported settings,
which is what every released Auto-AVSR model uses.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from mouthpiece.model.frontend import Conv3dResNet
from mouthpiece.model.layers import (
    ConvolutionModule,
    LayerNorm,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    RelPositionalEncoding,
    RelPositionMultiHeadedAttention,
)

# Per-layer cache for incremental decoding: one (B, L-1, D) tensor per layer.
Cache = list[Tensor]


class ConformerLayer(nn.Module):
    """Macaron FFN/2 -> rel-pos MHSA -> conv module -> FFN/2 -> final LayerNorm."""

    def __init__(self, size: int, heads: int, units: int, kernel: int, zero_triu: bool) -> None:
        super().__init__()
        self.self_attn = RelPositionMultiHeadedAttention(heads, size, zero_triu)
        self.feed_forward = PositionwiseFeedForward(size, units)
        self.feed_forward_macaron = copy.deepcopy(self.feed_forward)
        self.conv_module = ConvolutionModule(size, kernel)
        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)
        self.norm_ff_macaron = LayerNorm(size)
        self.norm_conv = LayerNorm(size)
        self.norm_final = LayerNorm(size)

    def forward(self, x: Tensor, pos_emb: Tensor, mask: Tensor | None) -> Tensor:
        x = x + 0.5 * self.feed_forward_macaron(self.norm_ff_macaron(x))
        h = self.norm_mha(x)
        x = x + self.self_attn.forward_rel(h, h, h, pos_emb, mask)
        x = x + self.conv_module(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward(self.norm_ff(x))
        out: Tensor = self.norm_final(x)
        return out


class ConformerEncoder(nn.Module):
    """Video frontend + Linear(512, adim) + rel-pos Conformer blocks."""

    def __init__(
        self,
        attention_dim: int = 768,
        attention_heads: int = 12,
        linear_units: int = 3072,
        num_blocks: int = 12,
        cnn_module_kernel: int = 31,
        zero_triu: bool = False,
    ) -> None:
        super().__init__()
        self.frontend = Conv3dResNet()
        self.embed = nn.Sequential(
            nn.Linear(512, attention_dim), RelPositionalEncoding(attention_dim)
        )
        self.encoders = nn.ModuleList(
            ConformerLayer(
                attention_dim, attention_heads, linear_units, cnn_module_kernel, zero_triu
            )
            for _ in range(num_blocks)
        )
        self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs: Tensor, mask: Tensor | None = None) -> Tensor:
        """xs (B, 1, T, H, W) -> (B, T, adim). mask: (B, 1, T) bool or None."""
        feats = self.frontend(xs)
        x, pos_emb = self.embed(feats)
        for layer in self.encoders:
            x = layer(x, pos_emb, mask)
        out: Tensor = self.after_norm(x)
        return out


class TransformerLayer(nn.Module):
    """Pre-norm self-attention + FFN block with optional incremental cache (used by the LM)."""

    def __init__(self, size: int, heads: int, units: int) -> None:
        super().__init__()
        self.size = size
        self.self_attn = MultiHeadedAttention(heads, size)
        self.feed_forward = PositionwiseFeedForward(size, units)
        self.norm_mha = LayerNorm(size)
        self.norm_ff = LayerNorm(size)

    def forward(self, x: Tensor, mask: Tensor | None, cache: Tensor | None = None) -> Tensor:
        h = self.norm_mha(x)
        if cache is None:
            q, residual, q_mask = h, x, mask
        else:
            q, residual = h[:, -1:], x[:, -1:]
            q_mask = None if mask is None else mask[:, -1:]
        x_new = residual + self.self_attn(q, h, h, q_mask)
        x_new = x_new + self.feed_forward(self.norm_ff(x_new))
        return x_new if cache is None else torch.cat([cache, x_new], dim=1)


class DecoderLayer(nn.Module):
    def __init__(self, size: int, heads: int, units: int) -> None:
        super().__init__()
        self.size = size
        self.self_attn = MultiHeadedAttention(heads, size)
        self.src_attn = MultiHeadedAttention(heads, size)
        self.feed_forward = PositionwiseFeedForward(size, units)
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Tensor | None,
        memory_kv: tuple[Tensor, Tensor],
        memory_mask: Tensor | None = None,
        cache: Tensor | None = None,
    ) -> Tensor:
        """memory_kv: src_attn keys/values already projected from the encoder output."""
        h = self.norm1(tgt)
        if cache is None:
            q, residual, q_mask = h, tgt, tgt_mask
        else:
            q, residual = h[:, -1:], tgt[:, -1:]
            q_mask = None if tgt_mask is None else tgt_mask[:, -1:]
        x = residual + self.self_attn(q, h, h, q_mask)
        x = x + self.src_attn.forward_kv(self.norm2(x), *memory_kv, memory_mask)
        x = x + self.feed_forward(self.norm3(x))
        return x if cache is None else torch.cat([cache, x], dim=1)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        odim: int,
        attention_dim: int = 768,
        attention_heads: int = 12,
        linear_units: int = 3072,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Embedding(odim, attention_dim), PositionalEncoding(attention_dim)
        )
        self.decoders = nn.ModuleList(
            DecoderLayer(attention_dim, attention_heads, linear_units) for _ in range(num_blocks)
        )
        self.after_norm = LayerNorm(attention_dim)
        self.output_layer = nn.Linear(attention_dim, odim)

    def memory_kv(self, memory: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Project the encoder output once per layer (B, H, T, d_k); reused at every step."""
        layers: list[DecoderLayer] = list(self.decoders)  # type: ignore[arg-type]
        return [layer.src_attn.project_kv(memory, memory) for layer in layers]

    def forward(
        self, tgt: Tensor, tgt_mask: Tensor | None, memory: Tensor, memory_mask: Tensor | None
    ) -> Tensor:
        """Full teacher-forced pass: logits (B, L, odim)."""
        x = self.embed(tgt)
        for layer, kv in zip(self.decoders, self.memory_kv(memory), strict=True):
            x = layer(x, tgt_mask, kv, memory_mask)
        out: Tensor = self.output_layer(self.after_norm(x))
        return out

    def forward_one_step(
        self,
        tgt: Tensor,
        tgt_mask: Tensor,
        memory_kv: list[tuple[Tensor, Tensor]],
        cache: Cache | None = None,
    ) -> tuple[Tensor, Cache]:
        """Log-probs for the next token (B, odim) and the updated per-layer cache."""
        x = self.embed(tgt)
        new_cache: Cache = []
        for i, (layer, kv) in enumerate(zip(self.decoders, memory_kv, strict=True)):
            x = layer(x, tgt_mask, kv, None, None if cache is None else cache[i])
            new_cache.append(x)
        logits = self.output_layer(self.after_norm(x[:, -1]))
        return torch.log_softmax(logits, dim=-1), new_cache
