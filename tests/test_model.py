from pathlib import Path

import numpy as np
import pytest
import torch

from mouthpiece.model.layers import RelPositionMultiHeadedAttention, subsequent_mask
from mouthpiece.model.lm import LMConfig, TransformerLM
from mouthpiece.model.tokenizer import Tokenizer
from mouthpiece.model.vsr import ModelLoadError, VSRConfig, VSRModel, load_vsr_model

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models/vsr/model.pth"
MODEL_JSON = ROOT / "models/vsr/model.json"
UNITS = ROOT / "models/vsr/unigram5000_units.txt"
SPM = ROOT / "models/vsr/unigram5000.model"

TINY = VSRConfig(odim=12, adim=32, aheads=4, eunits=64, elayers=2, dunits=64, dlayers=2,
                 cnn_module_kernel=3)


def tiny_model(seed: int = 0) -> VSRModel:
    torch.manual_seed(seed)
    return VSRModel(TINY).eval()


def test_tiny_model_forward_shapes() -> None:
    model = tiny_model()
    video = torch.randn(10, 88, 88)
    enc = model.encode(video)
    assert enc.shape == (10, 32)  # one encoder state per video frame (no subsampling)
    assert torch.isfinite(enc).all()
    logp = model.ctc.log_softmax(enc)
    assert logp.shape == (10, 12)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(10), atol=1e-5)
    ys = torch.tensor([[11, 3, 4]])
    logits = model.decoder(ys, subsequent_mask(3).unsqueeze(0), enc[None], None)
    assert logits.shape == (1, 3, 12)


def test_encoder_is_batch_and_length_consistent() -> None:
    model = tiny_model()
    a, b = torch.randn(8, 88, 88), torch.randn(8, 88, 88)
    with torch.no_grad():
        batched = model.encoder(torch.stack([a, b])[:, None])
    assert torch.allclose(batched[1], model.encode(b), atol=1e-5)


def test_decoder_incremental_matches_full() -> None:
    model = tiny_model()
    enc = model.encode(torch.randn(6, 88, 88))[None]
    dec = model.decoder
    ys = torch.tensor([[11, 2, 5, 7], [11, 3, 3, 1]])
    with torch.no_grad():
        full = torch.log_softmax(dec(ys, subsequent_mask(4).unsqueeze(0), enc, None), -1)
        kv = dec.memory_kv(enc)
        cache = None
        for i in range(1, 5):
            mask = subsequent_mask(i).unsqueeze(0)
            step, cache = dec.forward_one_step(ys[:, :i], mask, kv, cache)
            assert torch.allclose(step, full[:, i - 1], atol=1e-5)


def test_lm_incremental_matches_full() -> None:
    torch.manual_seed(0)
    lm = TransformerLM(LMConfig(n_vocab=12, layer=2, unit=32, att_unit=16, embed_unit=8,
                                head=2)).eval()
    ys = torch.tensor([[11, 4, 2, 9, 1]])
    with torch.no_grad():
        full, _ = lm.forward_one_step(ys)
        cache = None
        for i in range(1, 6):
            step, cache = lm.forward_one_step(ys[:, :i], cache)
        assert torch.allclose(step, full, atol=1e-5)
        assert torch.allclose(step.exp().sum(), torch.tensor(1.0), atol=1e-5)


def test_rel_shift_aligns_relative_positions() -> None:
    """Row i, column j must pick relative position i - j, i.e. pos_emb index T-1-i+j."""
    t = 5
    attn = RelPositionMultiHeadedAttention(1, 4)
    k = torch.arange(2 * t - 1, dtype=torch.float32)
    x = (10 * torch.arange(t, dtype=torch.float32)[:, None] + k[None, :])[None, None]
    out = attn.rel_shift(x)[0, 0]
    for i in range(t):
        for j in range(t):
            assert out[i, j] == 10 * i + (t - 1 - i + j)


def test_vsr_config_rejects_unsupported(tmp_path: Path) -> None:
    p = tmp_path / "model.json"
    p.write_text('[80, 41, {"transformer_input_layer": "conv1d"}]')
    with pytest.raises(ModelLoadError):
        VSRConfig.from_json(p, 5049)


def test_tokenizer_roundtrip(tmp_path: Path) -> None:
    units = tmp_path / "units.txt"
    units.write_text("<unk> 1\n▁HELLO 2\n▁WOR 3\nLD 4\n")
    tok = Tokenizer.from_files(units)
    assert tok.vocab_size == 6 and tok.blank == 0 and tok.eos == tok.sos == 5
    assert tok.decode([0, 2, 3, 4, 5]) == "HELLO WORLD"
    with pytest.raises(RuntimeError):
        tok.encode("hello")
    units.write_text("<unk> 1\nX 3\n")
    with pytest.raises(ValueError):
        Tokenizer.from_files(units)


@pytest.mark.skipif(not (UNITS.exists() and SPM.exists()), reason="tokenizer not downloaded")
def test_real_tokenizer() -> None:
    tok = Tokenizer.from_files(UNITS, SPM)
    assert tok.vocab_size == 5049
    ids = tok.encode("hello world")
    assert all(0 < i < tok.eos for i in ids)
    assert tok.decode(ids) == "HELLO WORLD"


@pytest.mark.skipif(not (CKPT.exists() and MODEL_JSON.exists()), reason="weights not downloaded")
def test_checkpoint_matches_architecture() -> None:
    sd = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
    odim = sd["ctc.ctc_lo.weight"].shape[0]
    assert odim == 5049
    with torch.device("meta"):
        model = VSRModel(VSRConfig.from_json(MODEL_JSON, odim))
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in sd.items()}
    assert ours == theirs
    # 12 conformer blocks, 6 decoder blocks, as documented.
    def blocks(prefix: str) -> int:
        return len({k.split(".")[2] for k in sd if k.startswith(prefix)})

    assert blocks("encoder.encoders.") == 12 and blocks("decoder.decoders.") == 6
    with pytest.raises(ModelLoadError):
        load_vsr_model(CKPT, MODEL_JSON, vocab_size=41)


def test_load_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        load_vsr_model(tmp_path / "nope.pth", MODEL_JSON)


def test_to_model_input_matches_recognizer_prepare() -> None:
    from mouthpiece.config import DecodeConfig
    from mouthpiece.decode.recognizer import Recognizer

    tok = Tokenizer([f"u{i}" for i in range(10)])
    rec = Recognizer(tiny_model(), tok, DecodeConfig())
    crops = np.full((3, 96, 96), 255, dtype=np.uint8)
    x = rec.prepare(crops)
    assert x.shape == (3, 88, 88)
    assert torch.allclose(x, torch.full_like(x, (1 - 0.421) / 0.165))
    with pytest.raises(ValueError):
        rec.prepare(np.zeros((3, 96, 96), dtype=np.float32))
