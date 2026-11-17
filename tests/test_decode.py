import itertools
import math

import numpy as np
import pytest
import torch
from torch import Tensor

from mouthpiece.config import DecodeConfig
from mouthpiece.decode.beam_search import BeamSearch, BeamSearchConfig, Hypothesis, end_detect
from mouthpiece.decode.closed_set import PhraseSet, attention_logprobs, ctc_logprobs, rank
from mouthpiece.decode.ctc_prefix import CTCPrefixScorer
from mouthpiece.decode.recognizer import Recognizer
from mouthpiece.model.tokenizer import Tokenizer
from mouthpiece.model.vsr import VSRConfig, VSRModel

# Toy vocabulary: 0 blank, 1 "a", 2 "b", 3 "c", 4 eos/sos.
V, BLANK, EOS = 5, 0, 4


def collapse(path: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    prev = None
    for s in path:
        if s != prev and s != BLANK:
            out.append(s)
        prev = s
    return tuple(out)


def brute_force(logp: Tensor) -> dict[tuple[int, ...], float]:
    """P(label sequence) for every sequence, by enumerating all CTC paths."""
    t = logp.size(0)
    probs: dict[tuple[int, ...], float] = {}
    p = logp.exp().numpy()
    for path in itertools.product(range(4), repeat=t):  # no eos in paths
        pr = float(np.prod([p[i, s] for i, s in enumerate(path)]))
        key = collapse(path)
        probs[key] = probs.get(key, 0.0) + pr
    return probs


def toy_ctc_logp(t: int = 5, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(t, V, generator=g) * 2
    logits[:, EOS] = -1e4  # eos is never emitted by CTC itself
    return torch.log_softmax(logits.double(), -1)


@pytest.mark.parametrize("prefix", [(1,), (2, 2), (1, 2, 1), (3, 1)])
def test_ctc_prefix_scores_match_brute_force(prefix: tuple[int, ...]) -> None:
    logp = toy_ctc_logp()
    probs = brute_force(logp)
    scorer = CTCPrefixScorer(logp, BLANK, EOS)
    state = scorer.init_state()
    total = 0.0
    last = torch.tensor([EOS])  # sos
    for i, tok in enumerate(prefix):
        out = scorer.score(last, i, state, None)
        total += float(out.scores[0, tok])
        # prefix probability = sum over all sequences starting with the prefix so far
        want = sum(v for k, v in probs.items() if k[: i + 1] == prefix[: i + 1])
        assert total == pytest.approx(math.log(want), abs=1e-6)
        state = CTCPrefixScorer.select(out, torch.tensor([0]), torch.tensor([tok]))
        last = torch.tensor([tok])
    # eos score completes the prefix into exactly that sequence
    out = scorer.score(last, len(prefix), state, None)
    final = total + float(out.scores[0, EOS])
    assert final == pytest.approx(math.log(probs[prefix]), abs=1e-6)
    assert out.scores[0, BLANK] < -1e9


def test_ctc_partial_scoring_matches_full() -> None:
    logp = toy_ctc_logp(seed=3)
    scorer = CTCPrefixScorer(logp, BLANK, EOS)
    state = scorer.init_state()
    full = scorer.score(torch.tensor([EOS]), 0, state, None)
    part = scorer.score(torch.tensor([EOS]), 0, state, torch.tensor([[2, 1]]))
    for tok in (1, 2, EOS):
        assert float(part.scores[0, tok]) == pytest.approx(float(full.scores[0, tok]))
    assert part.scores[0, 3] < -1e9  # not scored


def table_decoder(table: dict[tuple[int, ...], list[float]]):  # type: ignore[no-untyped-def]
    """Step function whose next-token distribution depends on the prefix (sos stripped)."""
    def step(ys: Tensor, cache: list[Tensor] | None) -> tuple[Tensor, list[Tensor]]:
        rows = []
        for y in ys.tolist():
            probs = table.get(tuple(y[1:]), [0.0, 0.1, 0.1, 0.1, 0.7])
            rows.append(torch.log(torch.tensor(probs) + 1e-12))
        return torch.stack(rows), [torch.zeros(ys.size(0), ys.size(1), 1)]
    return step


TABLE = {
    (): [0.0, 0.5, 0.4, 0.1, 0.0],
    (1,): [0.0, 0.1, 0.2, 0.1, 0.6],
    (2,): [0.0, 0.0, 0.0, 0.95, 0.05],
    (2, 3): [0.0, 0.0, 0.0, 0.0, 1.0],
    (1, 1): [0.0, 0.0, 0.0, 0.0, 1.0],
    (1, 2): [0.0, 0.0, 0.0, 0.0, 1.0],
}


def seq_logprob(seq: tuple[int, ...]) -> float:
    total = 0.0
    for i in range(len(seq) + 1):
        tok = seq[i] if i < len(seq) else EOS
        total += math.log(TABLE.get(seq[:i], [0.0, 0.1, 0.1, 0.1, 0.7])[tok] + 1e-12)
    return total


def test_beam_search_attention_only_finds_best_sequence() -> None:
    cfg = BeamSearchConfig(beam_size=4, ctc_weight=0.0, lm_weight=0.0)
    bs = BeamSearch(table_decoder(TABLE), V, EOS, EOS, cfg)
    hyps = bs(torch.device("cpu"), maxlen=6)
    # Greedy would take "a" (0.5) then eos -> 0.30; the best is "b c" -> 0.4*0.95*1.0 = 0.38.
    assert hyps[0].tokens == (2, 3)
    assert hyps[0].score == pytest.approx(math.log(0.38), abs=1e-6)
    assert hyps[1].tokens == (1,)
    scores = [h.score for h in hyps]
    assert scores == sorted(scores, reverse=True)
    for h in hyps[:3]:
        assert h.score == pytest.approx(seq_logprob(h.tokens), abs=1e-5)
        assert h.scores["decoder"] == pytest.approx(h.score, abs=1e-5)


def test_beam_size_one_is_greedy() -> None:
    cfg = BeamSearchConfig(beam_size=1, ctc_weight=0.0, lm_weight=0.0)
    hyps = BeamSearch(table_decoder(TABLE), V, EOS, EOS, cfg)(torch.device("cpu"), maxlen=6)
    assert hyps[0].tokens == (1,)


def test_ctc_weight_and_lm_change_the_winner() -> None:
    # CTC posteriors that clearly spell "a a": a, blank, a, blank
    logits = torch.full((4, V), -8.0)
    for t, s in enumerate([1, 0, 1, 0]):
        logits[t, s] = 0.0
    ctc = torch.log_softmax(logits, -1)
    cfg = BeamSearchConfig(beam_size=4, ctc_weight=0.9, lm_weight=0.0)
    hyps = BeamSearch(table_decoder(TABLE), V, EOS, EOS, cfg, ctc_logp=ctc)(
        torch.device("cpu"), maxlen=4)
    assert hyps[0].tokens == (1, 1)
    assert set(hyps[0].scores) == {"decoder", "ctc"}
    total = 0.1 * hyps[0].scores["decoder"] + 0.9 * hyps[0].scores["ctc"]
    assert hyps[0].score == pytest.approx(total, abs=1e-4)

    # An LM that loves "c" pulls the attention-only search towards (2, 3) -> ... -> c
    lm_table = {(): [0.0, 0.0, 0.0, 1.0, 0.0], (3,): [0.0, 0.0, 0.0, 0.0, 1.0]}
    cfg = BeamSearchConfig(beam_size=4, ctc_weight=0.0, lm_weight=1.0)
    hyps = BeamSearch(table_decoder(TABLE), V, EOS, EOS, cfg, lm=table_decoder(lm_table))(
        torch.device("cpu"), maxlen=4)
    assert hyps[0].tokens == (3,)


def test_length_bonus_favours_longer() -> None:
    # Every prefix: eos 0.7, each letter 0.1. Without a bonus the empty output wins.
    for bonus, length in ((0.0, 0), (5.0, 3)):  # 3 = maxlen - 1 (last step forces eos)
        cfg = BeamSearchConfig(beam_size=4, ctc_weight=0.0, lm_weight=0.0, length_bonus=bonus)
        hyps = BeamSearch(table_decoder({}), V, EOS, EOS, cfg)(torch.device("cpu"), maxlen=4)
        assert len(hyps[0].tokens) == length


def test_end_detect() -> None:
    assert not end_detect([], 5)
    good = Hypothesis((1, 2), -1.0)          # length 4
    hyps = [good] + [Hypothesis((1,) * n, -30.0) for n in (3, 4, 5)]  # lengths 5, 6, 7
    assert end_detect(hyps, 7)
    assert not end_detect(hyps + [Hypothesis((1,) * 4, -2.0)], 7)


def test_recognizer_end_to_end_tiny_model() -> None:
    torch.manual_seed(0)
    model = VSRModel(VSRConfig(odim=12, adim=32, aheads=4, eunits=64, elayers=2, dunits=64,
                               dlayers=2, cnn_module_kernel=3)).eval()
    tok = Tokenizer(["<unk>"] + [f"▁w{i}" for i in range(9)])
    rec = Recognizer(model, tok, DecodeConfig(beam_size=5, ctc_weight=0.3, lm_weight=0.0,
                                              top_k=3))
    mouths = np.random.default_rng(0).integers(0, 255, (12, 96, 96), dtype=np.uint8)
    t = rec.transcribe(mouths, preprocess_s=0.5)
    assert 1 <= len(t.nbest) <= 3
    assert t.text == t.nbest[0].text
    assert t.frames == 12 and t.device == "cpu"
    assert t.timing.total_s >= 0.5 and t.timing.encoder_s > 0
    assert [n.score for n in t.nbest] == sorted((n.score for n in t.nbest), reverse=True)
    assert rec.with_decode(beam_size=2).decode_cfg.beam_size == 2
    assert rec.decode_cfg.beam_size == 5


# --- closed-set (command mode) scoring ----------------------------------------


def test_closed_set_ctc_matches_brute_force() -> None:
    logp = toy_ctc_logp(t=5, seed=3)
    seqs = [(1,), (2, 2), (1, 2, 3), (3, 1, 3, 1, 3, 1)]   # last cannot fit in 5 frames
    probs = brute_force(logp)
    got = ctc_logprobs(logp, seqs, BLANK)
    for s, g in zip(seqs[:3], got[:3], strict=True):
        assert float(g) == pytest.approx(math.log(probs[s]), abs=1e-5)
    assert float(got[3]) == -math.inf


def _tiny_model() -> VSRModel:
    torch.manual_seed(0)
    return VSRModel(VSRConfig(odim=12, adim=32, aheads=4, eunits=64, elayers=2, dunits=64,
                              dlayers=2, cnn_module_kernel=3)).eval()


@torch.no_grad()
def test_closed_set_attention_matches_stepwise_decoding() -> None:
    model = _tiny_model()
    enc = torch.randn(9, 32)
    sos = eos = 11
    seqs = [(3,), (4, 5, 6), (7, 7)]
    got = attention_logprobs(model.decoder, enc, seqs, sos, eos)
    kv = model.decoder.memory_kv(enc.unsqueeze(0))
    for s, g in zip(seqs, got, strict=True):
        ys, total = [sos], 0.0
        for tok in (*s, eos):
            x = torch.tensor([ys])
            mask = torch.ones(len(ys), len(ys), dtype=torch.bool).tril().unsqueeze(0)
            logp, _ = model.decoder.forward_one_step(x, mask, kv)
            total += float(logp[0, tok])
            ys.append(tok)
        assert float(g) == pytest.approx(total, abs=1e-4)


def test_rank_combines_and_sorts() -> None:
    att = torch.tensor([-1.0, -3.0, -2.0])
    ctc = torch.tensor([-10.0, -1.0, -2.0])
    by_att = rank(att, None, ["a", "b", "c"], ctc_weight=0.5)
    assert [r.text for r in by_att] == ["a", "c", "b"]
    mixed = rank(att, ctc, ["a", "b", "c"], ctc_weight=0.5)
    assert [r.text for r in mixed] == ["b", "c", "a"]
    assert mixed[0].score == pytest.approx(-2.0) and mixed[0].att == -3.0


def test_phrase_set_build_dedupes() -> None:
    ps = PhraseSet.build(["copy that", "Copy that!", "", "go"], lambda s: [len(s)])
    assert ps.texts == ("copy that", "go") and ps.tokens == ((9,), (2,))
    with pytest.raises(ValueError):
        PhraseSet((), ())


def test_recognizer_closed_set_tiny_model() -> None:
    tok = Tokenizer(["<unk>"] + [f"▁w{i}" for i in range(9)])
    rec = Recognizer(_tiny_model(), tok, DecodeConfig(beam_size=5, ctc_weight=0.3,
                                                      lm_weight=0.0, top_k=3))
    phrases = PhraseSet(("w1", "w2 w3", "w4", "w5 w5"), ((2,), (3, 4), (5,), (6, 6)))
    mouths = np.random.default_rng(0).integers(0, 255, (12, 96, 96), dtype=np.uint8)
    t = rec.transcribe_closed_set(mouths, phrases, preprocess_s=0.1)
    assert len(t.nbest) == 3 and t.text == t.nbest[0].text
    assert {n.text for n in t.nbest} <= set(phrases.texts)
    assert [n.score for n in t.nbest] == sorted((n.score for n in t.nbest), reverse=True)
    enc = rec.encode(mouths)
    att = attention_logprobs(rec.model.decoder, enc, phrases.tokens, tok.sos, tok.eos)
    ctc = ctc_logprobs(rec.model.ctc.log_softmax(enc), phrases.tokens)
    best = int(torch.argmax(0.7 * att + 0.3 * ctc))
    assert t.text == phrases.texts[best]
