import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mouthpiece.cli import main
from mouthpiece.config import Config, TrainConfig
from mouthpiece.model.checkpoint import resolve_checkpoint
from mouthpiece.model.vsr import VSRConfig, VSRModel, load_vsr_model
from mouthpiece.record.manifest import ManifestEntry
from mouthpiece.train.augment import TrainAugment
from mouthpiece.train.dataset import IGNORE_ID, Batch
from mouthpiece.train.freeze import apply_freeze, set_train_mode
from mouthpiece.train.loss import compute_loss, decoder_io, encode_batch, target_mask
from mouthpiece.train.runs import (
    activate,
    new_run_dir,
    save_weights,
    should_activate,
    write_config,
)
from mouthpiece.train.trainer import (
    TrainError,
    build_optimizer,
    frame_batches,
    lr_factor,
    split_entries,
)

TINY = VSRConfig(odim=12, adim=32, aheads=4, eunits=64, elayers=4, dunits=64, dlayers=2,
                 cnn_module_kernel=3)


def tiny_model(seed: int = 0) -> VSRModel:
    torch.manual_seed(seed)
    model = VSRModel(TINY)
    # Non-trivial BatchNorm statistics, so a wrongly-trained BN would show up.
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            assert m.running_mean is not None
            m.running_mean.uniform_(-0.1, 0.1)
    return model.eval()


def tiny_batch(seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    videos = torch.randn(2, 12, 88, 88, generator=g)
    videos[1, 9:] = 0
    tokens = torch.tensor([[3, 4, 5], [6, 7, IGNORE_ID]])
    return Batch(videos, torch.tensor([12, 9]), tokens, torch.tensor([3, 2]), ["a", "b"],
                 ["x", "y"])


# --- freezing ---------------------------------------------------------------


def test_freeze_pattern() -> None:
    model = tiny_model()
    report = apply_freeze(model, trainable_blocks=1)
    enc = model.encoder

    def all_grad(m: nn.Module, flag: bool) -> bool:
        return all(p.requires_grad is flag for p in m.parameters())

    assert all_grad(enc.frontend, False) and all_grad(enc.embed, False)
    for block in list(enc.encoders)[:3]:
        assert all_grad(block, False)
    assert all_grad(enc.encoders[3], True)
    assert all_grad(enc.after_norm, True)
    assert all_grad(model.decoder, True) and all_grad(model.ctc, True)
    assert report.frozen_blocks == 3 and report.trainable_blocks == 1
    assert report.trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < report.trainable < report.total
    with pytest.raises(ValueError):
        apply_freeze(model, trainable_blocks=5)


def test_train_mode_keeps_batchnorm_and_frozen_parts_in_eval() -> None:
    model = tiny_model()
    report = apply_freeze(model, 2)
    set_train_mode(model, report.frozen_blocks)
    assert not model.encoder.frontend.training
    assert not model.encoder.encoders[1].training
    assert model.encoder.encoders[2].training and model.decoder.training
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bns and not any(m.training for m in bns)


# --- loss -------------------------------------------------------------------


def test_decoder_io_and_masks() -> None:
    tokens = torch.tensor([[3, 4, 5], [6, IGNORE_ID, IGNORE_ID]])
    ys_in, ys_out = decoder_io(tokens, torch.tensor([3, 1]), sos=11, eos=11)
    assert ys_in.tolist() == [[11, 3, 4, 5], [11, 6, 11, 11]]
    assert ys_out.tolist() == [[3, 4, 5, 11], [6, 11, -1, -1]]
    m = target_mask(torch.tensor([4, 2]), 4)
    assert m.shape == (2, 4, 4)
    assert m[0].tolist() == torch.ones(4, 4, dtype=torch.bool).tril().tolist()
    assert not m[1, 3, 2:].any() and m[1, 3, :2].all()  # padded keys dropped
    assert m[1].any(-1).all()                            # no fully masked row


def test_encode_batch_matches_encoder() -> None:
    model = tiny_model()
    b = tiny_batch()
    mask = (torch.arange(12)[None] < b.video_lengths[:, None]).unsqueeze(1)
    with torch.no_grad():
        ref = model.encoder(b.videos.unsqueeze(1), mask)
        out = encode_batch(model, b.videos, b.video_lengths, frozen_blocks=3)
    assert torch.allclose(out, ref, atol=1e-5)


def test_loss_is_finite_and_grads_only_on_trainable() -> None:
    model = tiny_model()
    report = apply_freeze(model, 1)
    set_train_mode(model, report.frozen_blocks)
    out = compute_loss(model, tiny_batch(), report.frozen_blocks, 0.1, 0.1)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.item() > 0
    assert out.ctc > 0 and out.att > 0 and 0.0 <= out.acc <= 1.0
    out.loss.backward()  # type: ignore[no-untyped-call]
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
        else:
            assert p.grad is None, name
    # attention-only and CTC-only weights also work
    assert torch.isfinite(compute_loss(model, tiny_batch(), 3, 0.0, 0.1).loss)
    assert torch.isfinite(compute_loss(model, tiny_batch(), 3, 1.0, 0.0).loss)


def test_optimizer_step_changes_only_trainable() -> None:
    model = tiny_model()
    report = apply_freeze(model, 1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    set_train_mode(model, report.frozen_blocks)
    out = compute_loss(model, tiny_batch(), report.frozen_blocks)
    opt.zero_grad()
    out.loss.backward()  # type: ignore[no-untyped-call]
    opt.step()
    after = model.state_dict()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed, "nothing was updated"
    assert changed <= trainable              # no frozen weight or BN statistic moved
    assert "decoder.output_layer.weight" in changed
    assert "encoder.encoders.3.feed_forward.w_1.weight" in changed


# --- augmentation, batching, schedule ---------------------------------------


def test_augment_shapes_and_ops() -> None:
    clip = np.random.default_rng(0).integers(1, 255, (50, 96, 96), dtype=np.uint8)
    aug = TrainAugment(88, True, 0.5, 10, 25, np.random.default_rng(1))
    out = aug(clip)
    assert out.shape == (50, 88, 88) and out.dtype == np.uint8

    center = TrainAugment(88, False, 0.0, 0, 25)(clip)
    assert np.array_equal(center, clip[:, 4:92, 4:92])
    flipped = TrainAugment(88, False, 1.0, 0, 25)(clip)
    assert np.array_equal(flipped, center[:, :, ::-1])

    masked = TrainAugment(88, False, 0.0, 10, 5, np.random.default_rng(0))(clip)
    zero_frames = int((masked.reshape(50, -1).max(1) == 0).sum())
    assert 0 < zero_frames < 50

    a = TrainAugment.from_config(TrainConfig(), 88, seed=3)(clip)
    b = TrainAugment.from_config(TrainConfig(), 88, seed=3)(clip)
    assert np.array_equal(a, b)


def test_frame_batches() -> None:
    lengths = [30, 100, 50, 250, 40, 60, 20]
    batches = frame_batches(lengths, 200, np.random.default_rng(0))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) * max(lengths[i] for i in b) <= 200 or len(b) == 1
    assert [3] in batches  # longer than the budget: alone


def test_lr_factor() -> None:
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0)


def _entry(i: int, split: str) -> ManifestEntry:
    return ManifestEntry(id=f"c{i}", text="hello", kind="command", tokens=[3], frames=20,
                         fps=25.0, detection_rate=1.0, created="", split=split,  # type: ignore[arg-type]
                         path=f"clips/c{i}")


def test_split_fallbacks() -> None:
    s = split_entries([_entry(0, "train"), _entry(1, "val")])
    assert s.real_val and len(s.train) == 1 and len(s.val) == 1 and s.warning is None
    s = split_entries([_entry(0, "train"), _entry(1, "train")])
    assert not s.real_val and s.val == s.train and s.warning
    s = split_entries([_entry(0, "val")])
    assert not s.real_val and s.train == s.val
    with pytest.raises(TrainError):
        split_entries([])


# --- checkpoints, activation, resolution ------------------------------------


def _write_model_json(path: Path, c: VSRConfig) -> None:
    args = {"transformer_input_layer": "conv3d", "transformer_encoder_attn_layer_type": "rel_mha",
            "rel_pos_type": "latest", "macaron_style": True, "use_cnn_module": True,
            "adim": c.adim, "aheads": c.aheads, "eunits": c.eunits, "elayers": c.elayers,
            "dunits": c.dunits, "dlayers": c.dlayers, "cnn_module_kernel": c.cnn_module_kernel}
    path.write_text(json.dumps([88, 41, args]))


def test_checkpoint_round_trip_strict(tmp_path: Path) -> None:
    model = tiny_model(seed=5)
    apply_freeze(model, 1)
    run = new_run_dir(tmp_path / "finetuned")
    save_weights(model, run / "best.pth")
    _write_model_json(tmp_path / "model.json", TINY)
    loaded = load_vsr_model(run / "best.pth", tmp_path / "model.json")  # strict=True inside
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k


def test_should_activate() -> None:
    assert should_activate(0.3, 0.5, True)
    assert not should_activate(0.5, 0.5, True)   # must strictly beat the baseline
    assert not should_activate(0.6, 0.5, True)
    assert not should_activate(0.1, 0.5, False)  # no real val split
    assert not should_activate(None, 0.5, True)


def test_activate_symlink(tmp_path: Path) -> None:
    out = tmp_path / "finetuned"
    run1, run2 = out / "20260101-000000", out / "20260102-000000"
    for r in (run1, run2):
        r.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        activate(run1)                            # no best.pth yet
    for r in (run1, run2):
        (r / "best.pth").write_bytes(b"x")
    link = activate(run1)
    assert link == out / "current" and link.is_symlink()
    assert not Path(link.readlink()).is_absolute()   # relative: survives moving the repo
    assert link.resolve() == run1.resolve()
    activate(run2)
    assert link.resolve() == run2.resolve()
    link.unlink()
    link.mkdir()
    with pytest.raises(FileExistsError):
        activate(run1)                            # never replace a real directory


def test_new_run_dir_unique(tmp_path: Path) -> None:
    from datetime import datetime

    now = datetime(2026, 1, 1, 12, 0, 0)
    a, b = new_run_dir(tmp_path, now), new_run_dir(tmp_path, now)
    assert a != b and a.is_dir() and b.is_dir()


def test_resolve_checkpoint_fallback(tmp_path: Path, cfg: Config) -> None:
    c = replace(cfg, root=tmp_path)
    active = resolve_checkpoint(c)
    assert not active.finetuned and active.path == tmp_path / c.model.checkpoint
    assert "pretrained" in active.describe(c)

    run = tmp_path / "models/finetuned/20260101-000000"
    run.mkdir(parents=True)
    link = tmp_path / c.model.finetuned  # type: ignore[operator]
    link.symlink_to(run.name, target_is_directory=True)
    assert not resolve_checkpoint(c).finetuned        # run without best.pth -> pretrained

    (run / "best.pth").write_bytes(b"x")
    (run / "metrics.json").write_text(json.dumps({"best_wer": 0.25, "baseline_wer": 0.5}))
    active = resolve_checkpoint(c)
    assert active.finetuned and active.path == link / "best.pth"
    assert active.run_dir == run.resolve()
    assert "25.0%" in active.describe(c) and "50.0%" in active.describe(c)

    off = replace(c, model=replace(c.model, finetuned=None))   # what --pretrained does
    assert not resolve_checkpoint(off).finetuned
    file_cfg = replace(c, model=replace(c.model, finetuned=str(run / "best.pth")))
    assert resolve_checkpoint(file_cfg).finetuned


def test_write_config_round_trips(tmp_path: Path, cfg: Config) -> None:
    import yaml

    write_config(tmp_path, cfg)
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert raw["train"]["trainable_blocks"] == cfg.train.trainable_blocks
    assert "root" not in raw
    p = tmp_path / "again.yaml"
    p.write_text((tmp_path / "config.yaml").read_text())
    from mouthpiece.config import load_config

    assert load_config(p).train == cfg.train


# --- CLI --------------------------------------------------------------------


def test_train_refuses_below_min_clips(tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(_entry(i, "train").to_json() + "\n" for i in range(3)))
    conf = tmp_path / "c.yaml"
    conf.write_text(f"data:\n  personal_dir: {tmp_path}\n  manifest: {manifest}\n"
                    "train:\n  min_clips: 5\n")
    assert main(["--config", str(conf), "train"]) == 1
    out = capsys.readouterr().out
    assert "min_clips" in out and "--smoke" in out


def test_train_config_validation(tmp_path: Path) -> None:
    from mouthpiece.config import ConfigError, load_config

    p = tmp_path / "c.yaml"
    p.write_text("train:\n  ctc_weight: 2.0\n")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text("train:\n  bogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(p)
