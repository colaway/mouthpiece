"""Subword tokenizer for the Auto-AVSR LRS3 models.

Token ids follow ESPnet's convention: 0 = <blank>, 1..N = lines of `unigram5000_units.txt`
(piece, id), N+1 = <sos/eos>. So the vocabulary is N + 2 = 5049. Decoding only needs the
units file; encoding text (for fine-tuning later) also needs the SentencePiece model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SPACE = "▁"  # SentencePiece word-boundary marker


class Tokenizer:
    blank = 0

    def __init__(self, units: Sequence[str], sp_model: Any | None = None) -> None:
        self.token_list: list[str] = ["<blank>", *units, "<eos>"]
        self.eos = self.sos = len(self.token_list) - 1
        self._index = {tok: i for i, tok in enumerate(self.token_list)}
        self.unk = self._index.get("<unk>", 1)
        self._sp = sp_model

    @property
    def vocab_size(self) -> int:
        return len(self.token_list)

    @classmethod
    def from_files(cls, units_path: str | Path, model_path: str | Path | None = None) -> Tokenizer:
        units: list[str] = []
        for n, line in enumerate(Path(units_path).read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) != 2 or int(parts[1]) != n:
                raise ValueError(f"{units_path}:{n}: expected '<piece> {n}', got {line!r}")
            units.append(parts[0])
        sp = None
        if model_path is not None and Path(model_path).exists():
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=str(model_path))
        return cls(units, sp)

    def ids_to_pieces(self, ids: Iterable[int]) -> list[str]:
        return [self.token_list[i] for i in ids if i not in (self.blank, self.eos)]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.ids_to_pieces(ids)).replace(SPACE, " ").strip()

    def encode(self, text: str) -> list[int]:
        """Text -> token ids (LRS3 transcripts are upper-case)."""
        if self._sp is None:
            raise RuntimeError("SentencePiece model not loaded; cannot encode text")
        pieces: list[str] = self._sp.encode(text.upper(), out_type=str)
        return [self._index.get(p, self.unk) for p in pieces]
