"""Command-line entry point: `mouthpiece <command>`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

from mouthpiece import __version__
from mouthpiece.config import Config, ConfigError, load_config

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from mouthpiece.decode.recognizer import Transcript
    from mouthpiece.model.tokenizer import Tokenizer


def _load_cfg(args: argparse.Namespace) -> Config:
    """Load the config; `--pretrained` disables the fine-tuned weights (model.finetuned)."""
    from dataclasses import replace

    cfg = load_config(args.config)
    if getattr(args, "pretrained", False):
        cfg = replace(cfg, model=replace(cfg.model, finetuned=None))
    return cfg


def _cmd_doctor(args: argparse.Namespace) -> int:
    from mouthpiece.doctor import exit_code, format_results, run_checks

    cfg = load_config(args.config)
    results = run_checks(cfg, probe_camera=args.probe_camera)
    print(format_results(results))
    code = exit_code(results)
    print("\nAll required checks passed." if code == 0 else "\nSome checks failed.")
    return code


def _cmd_preview(args: argparse.Namespace) -> int:
    from mouthpiece.app.preview_window import run_preview

    cfg = load_config(args.config)
    run_preview(cfg, mirror=not args.no_mirror)
    return 0


def _cmd_crop(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from mouthpiece.capture.camera import read_video
    from mouthpiece.preprocess.mouth_roi import CropError
    from mouthpiece.preprocess.pipeline import process_clip, save_clip

    cfg = load_config(args.config)
    t = time.perf_counter()
    frames, fps = read_video(args.video, target_fps=cfg.capture.fps)
    try:
        result = process_clip(cfg, frames, fps)
    except CropError as exc:
        print(f"Could not crop {args.video}: {exc}")
        return 1
    out = save_clip(Path(args.out), result)
    print(
        f"{len(frames)} frames @ {fps:.0f} fps, "
        f"face found in {result.detection_rate:.0%} of frames, "
        f"mouth crops {tuple(result.mouths.shape)} -> {out} ({time.perf_counter() - t:.1f}s)"
    )
    return 0


def _load_mouths(cfg: Config, path: str) -> tuple[npt.NDArray[np.generic], float, str]:
    """VIDEO, mouth.npy, or a clip directory containing mouth.npy -> (crops, preprocess_s, desc)."""
    import time
    from pathlib import Path

    import numpy as np

    p = Path(path)
    if p.is_dir():
        p = p / "mouth.npy"
    if p.suffix == ".npy":
        mouths = np.load(p, allow_pickle=False)
        return mouths, 0.0, f"{p} (pre-cropped)"
    from mouthpiece.capture.camera import read_video
    from mouthpiece.preprocess.pipeline import process_clip

    t = time.perf_counter()
    frames, fps = read_video(str(p), target_fps=cfg.capture.fps)
    result = process_clip(cfg, frames, fps)
    desc = f"{p} (face found in {result.detection_rate:.0%} of frames)"
    return result.mouths, time.perf_counter() - t, desc


def format_transcript_report(t: Transcript, fps: float, load_s: float | None = None) -> str:
    lines = [f"transcript: {t.text or '(empty)'}", "", f"top-{len(t.nbest)}:"]
    for i, n in enumerate(t.nbest, 1):
        parts = ", ".join(f"{k} {v:.2f}" for k, v in n.scores.items())
        lines.append(f"  {i}. {n.text or '(empty)':<40} score {n.score:8.2f}  ({parts})")
    dur = t.frames / fps
    tm = t.timing
    lines += ["", f"timing ({t.frames} frames = {dur:.2f} s of video, device {t.device}):"]
    if load_s is not None:
        lines.append(f"  model load  {load_s:6.2f} s  (not in total)")
    lines += [
        f"  preprocess  {tm.preprocess_s:6.2f} s",
        f"  encoder     {tm.encoder_s:6.2f} s",
        f"  decode      {tm.decode_s:6.2f} s",
        f"  total       {tm.total_s:6.2f} s  (real-time factor {tm.total_s / max(dur, 1e-6):.2f})",
    ]
    return "\n".join(lines)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    import time
    from dataclasses import replace

    from mouthpiece.decode.recognizer import Recognizer
    from mouthpiece.model.vsr import ModelLoadError
    from mouthpiece.preprocess.mouth_roi import CropError

    cfg = _load_cfg(args)
    overrides = {k: v for k, v in (("beam_size", args.beam_size), ("top_k", args.top_k))
                 if v is not None}
    if overrides:
        cfg = replace(cfg, decode=replace(cfg.decode, **overrides))
    try:
        mouths, pre_s, desc = _load_mouths(cfg, args.input)
    except (CropError, FileNotFoundError, ValueError) as exc:
        print(f"Could not read {args.input}: {exc}")
        return 1
    t = time.perf_counter()
    try:
        rec = Recognizer.from_config(cfg, args.device, use_lm=not args.no_lm,
                                     ctc_device=args.ctc_device)
    except (ModelLoadError, FileNotFoundError) as exc:
        print(f"Could not load model: {exc}")
        return 1
    load_s = time.perf_counter() - t
    print(f"input: {desc}, crops {tuple(mouths.shape)}")
    if rec.checkpoint is not None:
        print(f"weights: {rec.checkpoint.describe(cfg)}")
    print(f"model: {rec.device.type}, LM {'on' if rec.lm is not None else 'off'}, "
          f"beam {cfg.decode.beam_size}, ctc_weight {cfg.decode.ctc_weight}, "
          f"lm_weight {cfg.decode.lm_weight if rec.lm is not None else 0}\n")
    if args.warmup:
        rec.transcribe(mouths)
    try:
        result = rec.transcribe(mouths, preprocess_s=pre_s)
    except ValueError as exc:
        print(f"Could not transcribe: {exc}")
        return 1
    print(format_transcript_report(result, cfg.capture.fps, load_s))
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    from mouthpiece.app.live_window import run_live

    cfg = _load_cfg(args)
    run_live(cfg, mirror=not args.no_mirror, device=args.device, use_lm=not args.no_lm)
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from mouthpiece.app.menubar import run_app

    cfg = _load_cfg(args)
    if args.mode:
        cfg = replace(cfg, app=replace(cfg.app, mode=args.mode))
    run_app(cfg, device=args.device, use_lm=not args.no_lm, typing=not args.no_typing)
    return 0


def _cmd_type(args: argparse.Namespace) -> int:
    """Feed utterances through the typing stage, as if the app had recognised them."""
    import sys
    import time
    from dataclasses import replace

    from mouthpiece.app.permissions import accessibility_granted
    from mouthpiece.metrics import normalize_text
    from mouthpiece.typing.actions import COMMAND_ACTIONS
    from mouthpiece.typing.emitter import (
        Emitter,
        QuartzEmitter,
        RecordingEmitter,
        blocking_defer,
    )
    from mouthpiece.typing.output import TypingOutput

    cfg = _load_cfg(args)
    tcfg = cfg.typing
    if args.method:
        tcfg = replace(tcfg, method=args.method)
    rec = RecordingEmitter()
    emitter: Emitter = rec
    if not args.dry_run:
        if not accessibility_granted():
            print("Accessibility is not granted to the app running this command, so macOS "
                  "would drop the events. Grant it (System Settings → Privacy & Security → "
                  "Accessibility), restart that app, or use --dry-run.", file=sys.stderr)
            return 1
        emitter = QuartzEmitter(tcfg, defer=blocking_defer)
    out = TypingOutput(tcfg, emitter, blocking_defer, can_post=lambda: True,
                       on_signal=lambda sig: print(f"  (app would {sig} the camera)"),
                       log=lambda msg: print(msg.removeprefix("mouthpiece: ")))
    if not args.dry_run and args.delay > 0:
        print(f"typing in {args.delay:g} s — focus the target app…", flush=True)
        time.sleep(args.delay)

    class _Commit:
        def __init__(self, text: str, mode: str) -> None:
            self.text, self.mode = text, mode

    for u in args.utterances:
        mode = args.mode
        if mode == "auto":
            mode = "command" if normalize_text(u) in COMMAND_ACTIONS else "dictation"
        out(_Commit(u, mode))
    if args.dry_run:
        print("steps:  " + "\n        ".join(rec.calls))
        print(f"result: {rec.buffer!r}")
    return 0


def _load_tokenizer(cfg: Config) -> Tokenizer:
    from mouthpiece.model.tokenizer import Tokenizer

    m = cfg.model
    tok = Tokenizer.from_files(cfg.resolve(m.tokenizer_units), cfg.resolve(m.tokenizer))
    tok.encode("check")  # raises if the SentencePiece model is missing
    return tok


def _cmd_record(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from mouthpiece.app.recorder_window import run_record
    from mouthpiece.record.manifest import ManifestError, PersonalStore
    from mouthpiece.record.prompts import (
        PromptKind,
        PromptScheduler,
        builtin_prompts,
        load_prompt_file,
    )
    from mouthpiece.record.session import RecorderSession

    cfg = _load_cfg(args)
    r = cfg.record
    cfg = replace(cfg, record=replace(
        r,
        count=args.count or r.count,
        takes_per_phrase=args.takes_per_phrase or r.takes_per_phrase,
        baseline=r.baseline and not args.no_baseline,
        prompts=args.prompts or r.prompts,
    ))
    r = cfg.record
    kinds: list[PromptKind] = [] if args.prompts_only else (
        ["command", "dictation"] if args.kind == "all" else [args.kind])
    prompts = builtin_prompts(kinds)
    if r.prompts:
        try:
            prompts += load_prompt_file(cfg.resolve(r.prompts))
        except OSError as exc:
            print(f"Could not read prompts: {exc}")
            return 1
    if not prompts:
        print("No prompts to record (empty prompt file?).")
        return 1
    try:
        tok = _load_tokenizer(cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not load the tokenizer (run ./scripts/fetch_models.sh): {exc}")
        return 1
    store = PersonalStore.from_config(cfg)
    try:
        counts = store.phrase_counts()
    except ManifestError as exc:
        print(f"Manifest is corrupt, fix it before recording more: {exc}")
        return 1
    scheduler = PromptScheduler(prompts, counts, r.takes_per_phrase)
    session = RecorderSession(scheduler, store, tok.encode, r.count)
    left = len(scheduler.remaining())
    print(f"{len(scheduler.prompts)} phrases ({left} still under {r.takes_per_phrase} takes), "
          f"{sum(counts.values())} clips already in {store.manifest}")
    if session.prompt is None:
        print("Every phrase already has enough takes. Raise --takes-per-phrase or add --prompts.")
        return 0
    run_record(cfg, session, mirror=not args.no_mirror, device=args.device,
               use_lm=not args.no_lm)
    st = session.stats
    bw = st.baseline_wer
    print(f"kept {st.kept}, rejected {st.rejected}, redone {st.redone}, skipped {st.skipped}"
          + ("" if bw is None else f", baseline WER {bw:.1%}"))
    return 0


def _cmd_dataset_stats(args: argparse.Namespace) -> int:
    from mouthpiece.record.manifest import ManifestEntry, ManifestError, PersonalStore
    from mouthpiece.record.report import compute_stats, format_stats

    cfg = _load_cfg(args)
    store = PersonalStore.from_config(cfg)
    try:
        entries = store.entries()
    except ManifestError as exc:
        print(f"Manifest is corrupt: {exc}")
        return 1
    if not entries:
        print(f"No clips yet in {store.manifest}. Record some with `mouthpiece record`.")
        return 0
    rescore = None
    if args.rescore:
        from mouthpiece.decode.recognizer import Recognizer

        rec = Recognizer.from_config(cfg, args.device, use_lm=not args.no_lm)
        weights = rec.checkpoint.describe(cfg) if rec.checkpoint else "?"
        print(f"rescoring {len(entries)} clips on {rec.device.type} with {weights}…")

        def rescore(e: ManifestEntry) -> str:
            return rec.transcribe(store.load_mouths(e)).text

    stats = compute_stats(entries, rescore)
    print(format_stats(stats, top=args.top, target=cfg.record.takes_per_phrase))
    return 0


def _cmd_dataset_check(args: argparse.Namespace) -> int:
    from mouthpiece.record.manifest import ManifestError, PersonalStore
    from mouthpiece.record.report import check_dataset

    cfg = load_config(args.config)
    store = PersonalStore.from_config(cfg)
    try:
        entries = store.entries()
    except ManifestError as exc:
        print(f"❌ {exc}")
        return 1
    encode = None
    try:
        encode = _load_tokenizer(cfg).encode
    except (OSError, RuntimeError, ValueError):
        print("⚠️  tokenizer unavailable, skipping token re-encoding check")
    errors, warnings = check_dataset(store.root, entries, encode, cfg.preprocess.roi_size)
    for w in warnings:
        print(f"⚠️  {w}")
    for e in errors:
        print(f"❌ {e}")
    print(f"{len(entries)} clips, {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from mouthpiece.model.vsr import ModelLoadError
    from mouthpiece.record.manifest import ManifestError
    from mouthpiece.train.trainer import TrainError, TrainOptions, run_training

    cfg = load_config(args.config)
    if args.epochs is not None and args.epochs < 1:
        print("--epochs must be >= 1")
        return 1
    opts = TrainOptions(epochs=args.epochs, force=args.force, smoke=args.smoke,
                        activate=not args.no_activate, device=args.device,
                        use_lm=not args.no_lm)
    try:
        result = run_training(cfg, opts)
    except (TrainError, ManifestError, ModelLoadError, FileNotFoundError) as exc:
        print(f"Cannot train: {exc}")
        return 1
    best = "n/a" if result.best_wer is None else f"{result.best_wer:.1%}"
    print(f"\nbaseline WER {result.baseline_wer:.1%}, best WER {best} "
          f"(epoch {result.best_epoch}), activated: {'yes' if result.activated else 'no'}")
    print(f"run: {result.run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mouthpiece", description=__doc__)
    parser.add_argument("--version", action="version", version=f"mouthpiece {__version__}")
    parser.add_argument("--config", help="Path to YAML config (default: configs/mouthpiece.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check camera permission, MPS, and model files")
    doctor.add_argument(
        "--probe-camera", action="store_true",
        help="Open the camera and grab a frame (may trigger the macOS permission prompt)",
    )
    doctor.set_defaults(func=_cmd_doctor)

    preview = sub.add_parser("preview", help="Live debug window: landmarks, mouth crop, FPS")
    preview.add_argument("--no-mirror", action="store_true", help="Show the unmirrored camera view")
    preview.set_defaults(func=_cmd_preview)

    crop = sub.add_parser("crop", help="Preprocess a video file into mouth crops")
    crop.add_argument("video", help="Input video file")
    crop.add_argument("--out", required=True, help="Output directory")
    crop.set_defaults(func=_cmd_crop)

    tr = sub.add_parser("transcribe", help="Lip-read a video file or mouth crops (mouth.npy)")
    tr.add_argument("input", help="Video file, mouth.npy, or a clip folder containing mouth.npy")
    tr.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    tr.add_argument("--ctc-device", choices=["auto", "cpu"], default="auto",
                    help="Where CTC prefix scoring runs (auto = with the model)")
    tr.add_argument("--no-lm", action="store_true", help="Decode without the language model")
    tr.add_argument("--beam-size", type=int, help="Override decode.beam_size")
    tr.add_argument("--top-k", type=int, help="Override decode.top_k")
    tr.add_argument("--pretrained", action="store_true",
                    help="Use the pretrained weights even if a fine-tuned run is active")
    tr.add_argument("--warmup", action="store_true",
                    help="Run once before timing (excludes one-off MPS kernel compilation)")
    tr.set_defaults(func=_cmd_transcribe)

    live = sub.add_parser("live", help="Camera window: hold Space to talk, release to transcribe")
    live.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    live.add_argument("--no-lm", action="store_true", help="Decode without the language model")
    live.add_argument("--no-mirror", action="store_true", help="Show the unmirrored camera view")
    live.add_argument("--pretrained", action="store_true",
                      help="Use the pretrained weights even if a fine-tuned run is active")
    live.set_defaults(func=_cmd_live)

    app = sub.add_parser("app", help="Menu-bar app: hold the hotkey to talk, pick from the top 3")
    app.add_argument("--mode", choices=["command", "dictation"], help="Override app.mode")
    app.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    app.add_argument("--no-lm", action="store_true", help="Dictation without the language model")
    app.add_argument("--no-typing", action="store_true",
                     help="Only print commits; don't type into other apps")
    app.add_argument("--pretrained", action="store_true",
                     help="Use the pretrained weights even if fine-tuned ones are active")
    app.set_defaults(func=_cmd_app)

    ty = sub.add_parser("type", help="Run text through the typing stage (no camera needed)")
    ty.add_argument("utterances", nargs="+",
                    help="Utterances in order, e.g. 'HELLO THERE' period 'HOW ARE YOU'")
    ty.add_argument("--mode", choices=["auto", "command", "dictation"], default="auto",
                    help="auto: known command phrases are commands, the rest dictation")
    ty.add_argument("--dry-run", action="store_true",
                    help="Print the planned keystrokes and resulting text; type nothing")
    ty.add_argument("--delay", type=float, default=3.0,
                    help="Seconds to wait before typing, to focus the target app")
    ty.add_argument("--method", choices=["auto", "keys", "paste"], help="Override typing.method")
    ty.set_defaults(func=_cmd_type)

    rec = sub.add_parser("record", help="Record your own mouthed phrases as training data")
    rec.add_argument("--prompts", help="Extra prompt file, one phrase per line")
    rec.add_argument("--prompts-only", action="store_true",
                     help="Only use --prompts / record.prompts, not the built-in phrases")
    rec.add_argument("--kind", choices=["all", "command", "dictation"], default="all",
                     help="Which built-in phrase set to use")
    rec.add_argument("--count", type=int, help="Takes to keep this session (record.count)")
    rec.add_argument("--takes-per-phrase", type=int, help="Override record.takes_per_phrase")
    rec.add_argument("--no-baseline", action="store_true",
                     help="Don't transcribe takes with the current model")
    rec.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    rec.add_argument("--no-lm", action="store_true", help="Baseline without the language model")
    rec.add_argument("--no-mirror", action="store_true", help="Show the unmirrored camera view")
    rec.add_argument("--pretrained", action="store_true",
                     help="Use the pretrained weights even if a fine-tuned run is active")
    rec.set_defaults(func=_cmd_record)

    ds = sub.add_parser("dataset", help="Inspect the personal dataset")
    ds_sub = ds.add_subparsers(dest="dataset_command", required=True)
    stats = ds_sub.add_parser("stats", help="Clip counts, minutes, per-phrase takes, WER")
    stats.add_argument("--rescore", action="store_true",
                       help="Re-transcribe every clip with the current model for a fresh WER")
    stats.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    stats.add_argument("--no-lm", action="store_true", help="Rescore without the language model")
    stats.add_argument("--top", type=int, default=15, help="Phrases to list")
    stats.add_argument("--pretrained", action="store_true",
                       help="Use the pretrained weights even if a fine-tuned run is active")
    stats.set_defaults(func=_cmd_dataset_stats)
    check = ds_sub.add_parser("check", help="Verify manifest and clip files are consistent")
    check.set_defaults(func=_cmd_dataset_check)

    train = sub.add_parser("train", help="Fine-tune the model on your recordings")
    train.add_argument("--epochs", type=int, help="Override train.epochs")
    train.add_argument("--force", action="store_true",
                       help="Train even with fewer than train.min_clips clips")
    train.add_argument("--smoke", action="store_true",
                       help="Quick pipeline test: 1 epoch, few steps, never activated")
    train.add_argument("--no-activate", action="store_true",
                       help="Don't point models/finetuned/current at this run")
    train.add_argument("--device", choices=["auto", "mps", "cpu"], help="Override model.device")
    train.add_argument("--no-lm", action="store_true", help="Decode val without the LM")
    train.set_defaults(func=_cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        parser.error(str(exc))
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
