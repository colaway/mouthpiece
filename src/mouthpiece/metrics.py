"""Text normalization and word error rate (LRS3-style: upper-case words, no punctuation)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_DROP = re.compile(r"[^A-Z0-9' ]+")


def normalize_text(text: str) -> str:
    """Upper-case, punctuation removed (apostrophes kept), single spaces."""
    return " ".join(_DROP.sub(" ", text.upper().replace("’", "'")).split())


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference word count) after normalization."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return edit_distance(r, h), len(r)


def wer(ref: str, hyp: str) -> float:
    errs, n = word_errors(ref, hyp)
    return errs / n if n else float(errs > 0)


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    """Total word edits / total reference words over (ref, hyp) pairs."""
    errs = words = 0
    for ref, hyp in pairs:
        e, n = word_errors(ref, hyp)
        errs, words = errs + e, words + n
    return errs / words if words else 0.0
