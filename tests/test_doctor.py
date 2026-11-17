from dataclasses import replace
from pathlib import Path

from mouthpiece.cli import main
from mouthpiece.config import load_config
from mouthpiece.doctor import (
    CheckResult,
    Status,
    check_camera_permission,
    check_model_files,
    check_tokenizer,
    check_torch_device,
    exit_code,
    format_results,
)


def test_camera_permission_states() -> None:
    assert check_camera_permission(lambda: 3).status is Status.OK
    assert check_camera_permission(lambda: 0).status is Status.WARN
    assert check_camera_permission(lambda: 2).status is Status.FAIL
    assert check_camera_permission(lambda: 1).status is Status.FAIL
    assert check_camera_permission(lambda: None).status is Status.WARN


def test_model_files_missing_then_present(tmp_path: Path) -> None:
    cfg = replace(load_config(), root=tmp_path)
    results = check_model_files(cfg)
    by_name = {r.name: r.status for r in results}
    assert by_name["model checkpoint"] is Status.WARN
    assert by_name["face landmarker"] is Status.FAIL  # needed from Phase 1 on

    for rel in (cfg.model.checkpoint, cfg.model.model_config, cfg.model.tokenizer,
                cfg.preprocess.face_landmarker_model, cfg.preprocess.mean_face):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    units = tmp_path / cfg.model.tokenizer_units
    units.write_text("".join(f"P{i} {i}\n" for i in range(1, 5048)))
    results = check_model_files(cfg)
    required = [r for r in results if r.name != "language model"]
    assert all(r.status is Status.OK for r in required)
    assert any(r.name == "tokenizer vocab" for r in results)


def test_tokenizer_vocab_check(tmp_path: Path) -> None:
    units = tmp_path / "units.txt"
    units.write_text("A 1\nB 2\n")
    assert check_tokenizer(units).status is Status.FAIL  # 4 tokens, model wants 5049
    units.write_text("A 1\nB 7\n")
    assert check_tokenizer(units).status is Status.FAIL
    assert check_tokenizer(units, expected_vocab=4).status is Status.FAIL
    units.write_text("A 1\nB 2\n")
    assert check_tokenizer(units, expected_vocab=4).status is Status.OK


def test_torch_device_cpu_forced() -> None:
    assert check_torch_device("cpu").status is Status.OK


def test_exit_code_and_format() -> None:
    results = [
        CheckResult("a", Status.OK, "fine"),
        CheckResult("b", Status.WARN, "meh", "do x"),
    ]
    assert exit_code(results) == 0
    assert "do x" in format_results(results)
    assert exit_code([*results, CheckResult("c", Status.FAIL, "bad")]) == 1


def test_cli_doctor_runs(capsys: object) -> None:
    code = main(["doctor"])
    assert code in (0, 1)
