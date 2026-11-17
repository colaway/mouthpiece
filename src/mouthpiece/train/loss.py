"""Training forward pass and the Auto-AVSR hybrid loss.

loss = ctc_weight * CTC + (1 - ctc_weight) * label-smoothed attention cross-entropy, each
summed over the batch and divided by the batch size (ESPnet's normalization). The decoder is
teacher-forced on `[sos] + tokens` and predicts `tokens + [eos]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mouthpiece.model.vsr import VSRModel
from mouthpiece.train.dataset import IGNORE_ID, Batch


@dataclass(frozen=True)
class LossOutput:
    loss: Tensor          # scalar, differentiable
    ctc: float
    att: float
    acc: float            # attention decoder token accuracy (teacher-forced)


def length_mask(lengths: Tensor, max_len: int) -> Tensor:
    """(B,) lengths -> (B, max_len) bool, True for real positions."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def encode_batch(model: VSRModel, videos: Tensor, lengths: Tensor, frozen_blocks: int) -> Tensor:
    """(B, T, H, W) padded videos -> (B, T, adim). The frontend, embed and the lower
    `frozen_blocks` Conformer blocks run without autograd (they are frozen), which keeps
    activation memory down to the trained top of the encoder."""
    enc = model.encoder
    mask = length_mask(lengths, videos.size(1)).unsqueeze(1)  # (B, 1, T)
    blocks = list(enc.encoders)
    with torch.no_grad():
        x, pos_emb = enc.embed(enc.frontend(videos.unsqueeze(1)))
        for block in blocks[:frozen_blocks]:
            x = block(x, pos_emb, mask)
    for block in blocks[frozen_blocks:]:
        x = block(x, pos_emb, mask)
    out: Tensor = enc.after_norm(x)
    return out


def decoder_io(tokens: Tensor, lengths: Tensor, sos: int, eos: int) -> tuple[Tensor, Tensor]:
    """Padded targets (IGNORE_ID) -> decoder input `[sos]+y` (eos-padded) and output `y+[eos]`
    (IGNORE_ID-padded), both (B, L+1)."""
    b, lmax = tokens.shape
    ys_in = torch.full((b, lmax + 1), eos, dtype=torch.long, device=tokens.device)
    ys_out = torch.full((b, lmax + 1), IGNORE_ID, dtype=torch.long, device=tokens.device)
    ys_in[:, 0] = sos
    for i, n in enumerate(lengths.tolist()):
        ys_in[i, 1 : n + 1] = tokens[i, :n]
        ys_out[i, :n] = tokens[i, :n]
        ys_out[i, n] = eos
    return ys_in, ys_out


def target_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Causal + key-padding mask (B, L, L) for decoder self-attention; True = attend."""
    causal = torch.ones(max_len, max_len, dtype=torch.bool, device=lengths.device).tril()
    return causal[None] & length_mask(lengths, max_len)[:, None, :]


def ctc_loss(model: VSRModel, enc: Tensor, enc_lengths: Tensor, batch: Batch) -> Tensor:
    """Summed CTC loss / batch size. Computed on CPU: MPS has no CTC kernel."""
    logp = model.ctc.log_softmax(enc).transpose(0, 1).float().cpu()  # (T, B, V)
    targets = torch.cat([t[:n] for t, n in zip(batch.tokens, batch.token_lengths, strict=True)])
    loss = F.ctc_loss(logp, targets.cpu(), enc_lengths.cpu(), batch.token_lengths.cpu(),
                      blank=model.blank, reduction="sum", zero_infinity=True)
    return (loss / len(batch)).to(enc.device)


def compute_loss(
    model: VSRModel,
    batch: Batch,
    frozen_blocks: int,
    ctc_weight: float = 0.1,
    label_smoothing: float = 0.1,
) -> LossOutput:
    """Hybrid CTC/attention loss for a batch already on the model's device."""
    enc = encode_batch(model, batch.videos, batch.video_lengths, frozen_blocks)
    b = len(batch)
    ctc = ctc_loss(model, enc, batch.video_lengths, batch) if ctc_weight > 0 else enc.new_zeros(())

    ys_in, ys_out = decoder_io(batch.tokens, batch.token_lengths, model.sos, model.eos)
    in_lengths = batch.token_lengths + 1
    memory_mask = length_mask(batch.video_lengths, enc.size(1)).unsqueeze(1)
    logits = model.decoder(ys_in, target_mask(in_lengths, ys_in.size(1)), enc, memory_mask)
    att = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys_out.reshape(-1),
                          ignore_index=IGNORE_ID, label_smoothing=label_smoothing,
                          reduction="sum") / b
    valid = ys_out != IGNORE_ID
    acc = ((logits.argmax(-1) == ys_out) & valid).sum().item() / max(1, int(valid.sum().item()))

    loss = ctc_weight * ctc + (1.0 - ctc_weight) * att
    return LossOutput(loss, float(ctc.item()), float(att.item()), float(acc))
