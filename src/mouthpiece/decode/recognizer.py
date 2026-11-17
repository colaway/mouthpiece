"""End-to-end recognizer: mouth crops -> encoder -> joint CTC/attention(/LM) beam search."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from mouthpiece.config import Config, DecodeConfig
from mouthpiece.decode.beam_search import BeamSearch, BeamSearchConfig, Cache, Hypothesis
from mouthpiece.decode.closed_set import (
    PhraseSet,
    RankedPhrase,
    attention_logprobs,
    ctc_logprobs,
    rank,
)
from mouthpiece.model.checkpoint import ActiveCheckpoint, resolve_checkpoint
from mouthpiece.model.device import resolve_device, synchronize
from mouthpiece.model.lm import TransformerLM, load_lm
from mouthpiece.model.tokenizer import Tokenizer
from mouthpiece.model.vsr import VSRModel, load_vsr_model
from mouthpiece.preprocess.mouth_roi import to_model_input


@dataclass(frozen=True)
class Timing:
    preprocess_s: float = 0.0
    encoder_s: float = 0.0
    decode_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.preprocess_s + self.encoder_s + self.decode_s


@dataclass(frozen=True)
class NBestEntry:
    text: str
    score: float
    scores: dict[str, float]


@dataclass(frozen=True)
class Transcript:
    text: str
    nbest: list[NBestEntry]
    frames: int
    device: str
    timing: Timing = field(default_factory=Timing)


CtcDevice = Literal["auto", "cpu", "model"]


class Recognizer:
    def __init__(
        self,
        model: VSRModel,
        tokenizer: Tokenizer,
        decode: DecodeConfig,
        lm: TransformerLM | None = None,
        input_size: int = 88,
        mean: float = 0.421,
        std: float = 0.165,
        ctc_device: CtcDevice = "auto",
    ) -> None:
        self.model, self.tokenizer, self.lm = model, tokenizer, lm
        self.decode_cfg = decode
        self.input_size, self.mean, self.std = input_size, mean, std
        self.device = next(model.parameters()).device
        # CTC prefix scoring is a long chain of tiny ops. Measured on an M1 Pro, keeping it on
        # MPS next to the decoder is still a bit faster than CPU, so "auto" means the model's
        # device; "cpu" forces CPU (e.g. to work around an MPS kernel issue).
        self.ctc_device = torch.device("cpu") if ctc_device == "cpu" else self.device
        self.checkpoint: ActiveCheckpoint | None = None   # set by from_config

    @classmethod
    def from_config(
        cls, cfg: Config, device: str | None = None, use_lm: bool = True,
        ctc_device: CtcDevice = "auto",
    ) -> Recognizer:
        dev = resolve_device(device or cfg.model.device)
        m = cfg.model
        tok = Tokenizer.from_files(cfg.resolve(m.tokenizer_units), cfg.resolve(m.tokenizer))
        # Fine-tuned weights (models/finetuned/current) when activated, else pretrained.
        ckpt = resolve_checkpoint(cfg)
        model = load_vsr_model(ckpt.path, cfg.resolve(m.model_config), dev, tok.vocab_size)
        lm = None
        if use_lm and m.language_model and cfg.decode.lm_weight > 0:
            lm_path = cfg.resolve(m.language_model)
            if lm_path.exists():
                lm_conf = m.language_model_config or str(lm_path.with_suffix(".json"))
                lm = load_lm(lm_path, cfg.resolve(lm_conf), dev)
        p = cfg.preprocess
        rec = cls(model, tok, cfg.decode, lm, p.model_input_size, p.normalize_mean,
                  p.normalize_std, ctc_device)
        rec.checkpoint = ckpt
        return rec

    def with_decode(self, **changes: float | int) -> Recognizer:
        """Same loaded weights, different decode settings."""
        new = object.__new__(Recognizer)
        new.__dict__.update(self.__dict__)
        new.decode_cfg = replace(self.decode_cfg, **changes)  # type: ignore[arg-type]
        return new

    def prepare(self, mouths: npt.NDArray[np.generic]) -> Tensor:
        """(T, 96, 96) uint8 crops -> normalized (T, 88, 88); float input passes through."""
        if mouths.ndim != 3:
            raise ValueError(f"expected (T, H, W) mouth crops, got shape {mouths.shape}")
        if mouths.dtype == np.uint8:
            crops = np.asarray(mouths, dtype=np.uint8)
            x = to_model_input(crops, self.input_size, self.mean, self.std)
        else:
            x = np.asarray(mouths, dtype=np.float32)
            if x.shape[1:] != (self.input_size, self.input_size):
                raise ValueError(f"float input must be (T, {self.input_size}, {self.input_size})")
        return torch.from_numpy(np.ascontiguousarray(x))

    @torch.no_grad()
    def encode(self, mouths: npt.NDArray[np.generic]) -> Tensor:
        enc = self.model.encode(self.prepare(mouths))
        synchronize(self.device)
        return enc

    @torch.no_grad()
    def search(self, enc: Tensor) -> list[Hypothesis]:
        d = self.decode_cfg
        bs_cfg = BeamSearchConfig(d.beam_size, d.ctc_weight, d.lm_weight, d.length_bonus)
        dec = self.model.decoder
        memory_kv = dec.memory_kv(enc.unsqueeze(0))

        def decoder_step(ys: Tensor, cache: Cache | None) -> tuple[Tensor, Cache]:
            mask = torch.ones(ys.size(1), ys.size(1), dtype=torch.bool, device=ys.device).tril()
            return dec.forward_one_step(ys, mask.unsqueeze(0), memory_kv, cache)

        lm_step = None
        if self.lm is not None and d.lm_weight > 0:
            lm_step = self.lm.forward_one_step
        ctc_logp = None
        if d.ctc_weight > 0:
            ctc_logp = self.model.ctc.log_softmax(enc).to(self.ctc_device)
        search = BeamSearch(
            decoder_step, self.tokenizer.vocab_size, self.tokenizer.sos, self.tokenizer.eos,
            bs_cfg, ctc_logp=ctc_logp, lm=lm_step, blank=self.tokenizer.blank,
        )
        return search(self.device, maxlen=enc.size(0))

    def phrase_set(self, texts: Iterable[str]) -> PhraseSet:
        """Tokenize a closed set of phrases (command mode) once, up front."""
        return PhraseSet.build(texts, self.tokenizer.encode)

    @torch.no_grad()
    def rank_phrases(self, enc: Tensor, phrases: PhraseSet) -> list[RankedPhrase]:
        w = self.decode_cfg.ctc_weight
        att = attention_logprobs(self.model.decoder, enc, phrases.tokens,
                                 self.tokenizer.sos, self.tokenizer.eos)
        ctc = None
        if w > 0:
            ctc = ctc_logprobs(self.model.ctc.log_softmax(enc), phrases.tokens,
                               self.tokenizer.blank)
        return rank(att, ctc, phrases.texts, w)

    def transcribe_closed_set(
        self, mouths: npt.NDArray[np.generic], phrases: PhraseSet, preprocess_s: float = 0.0
    ) -> Transcript:
        """Command mode: the best `top_k` phrases from `phrases` (always valid commands)."""
        t0 = time.perf_counter()
        enc = self.encode(mouths)
        t1 = time.perf_counter()
        ranked = self.rank_phrases(enc, phrases)
        synchronize(self.device)
        t2 = time.perf_counter()
        nbest = [NBestEntry(r.text, r.score, {"decoder": r.att, "ctc": r.ctc})
                 for r in ranked[: max(1, self.decode_cfg.top_k)]]
        return Transcript(
            text=nbest[0].text,
            nbest=nbest,
            frames=int(mouths.shape[0]),
            device=str(self.device),
            timing=Timing(preprocess_s, t1 - t0, t2 - t1),
        )

    def transcribe(
        self, mouths: npt.NDArray[np.generic], preprocess_s: float = 0.0
    ) -> Transcript:
        t0 = time.perf_counter()
        enc = self.encode(mouths)
        t1 = time.perf_counter()
        hyps = self.search(enc)
        synchronize(self.device)
        t2 = time.perf_counter()
        nbest = [
            NBestEntry(self.tokenizer.decode(h.tokens), h.score, h.scores)
            for h in hyps[: max(1, self.decode_cfg.top_k)]
        ]
        return Transcript(
            text=nbest[0].text if nbest else "",
            nbest=nbest,
            frames=int(mouths.shape[0]),
            device=str(self.device),
            timing=Timing(preprocess_s, t1 - t0, t2 - t1),
        )
