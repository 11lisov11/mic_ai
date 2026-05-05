import json
import subprocess

import pytest

from tools import check_air56_unoq_firmware_static
from tools import check_air56_unoq_coverage_gate
from tools import run_air56_unoq_deploy_smoke


def test_static_compile_main_uses_compiler_and_temp_files(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sys.argv", ["check_air56_unoq_firmware_static.py", "--compiler", "fake-g++"])
    monkeypatch.setattr(check_air56_unoq_firmware_static.shutil, "which", lambda name: str(tmp_path / name))
    monkeypatch.setattr(check_air56_unoq_firmware_static.subprocess, "run", fake_run)

    assert check_air56_unoq_firmware_static.main() == 0
    assert calls
    cmd, check = calls[0]
    assert check is True
    assert "-DAIR56_UNOQ_USE_MOCK_HW=1" in cmd
    assert "air56_unoq_static_compile.o" in cmd[-1]


def test_static_compile_main_fails_without_compiler(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["check_air56_unoq_firmware_static.py", "--compiler", "missing-g++"])
    monkeypatch.setattr(check_air56_unoq_firmware_static.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit, match="compiler not found"):
        check_air56_unoq_firmware_static.main()


def test_deploy_smoke_run_step_dry_run() -> None:
    step = run_air56_unoq_deploy_smoke._run_step("dry", ["cmd"], dry_run=True)

    assert step.passed
    assert step.returncode == 0
    assert step.elapsed_s == 0.0


def test_deploy_smoke_run_step_reports_failure(monkeypatch) -> None:
    def fake_run(command, cwd):
        return subprocess.CompletedProcess(command, 5)

    monkeypatch.setattr(run_air56_unoq_deploy_smoke.subprocess, "run", fake_run)
    step = run_air56_unoq_deploy_smoke._run_step("fail", ["cmd"], dry_run=False)

    assert not step.passed
    assert step.returncode == 5


def test_deploy_smoke_cli_dry_run_writes_json(tmp_path, monkeypatch, capsys) -> None:
    out_json = tmp_path / "deploy_smoke.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_air56_unoq_deploy_smoke.py",
            "--dry-run",
            "--out-json",
            str(out_json),
        ],
    )

    assert run_air56_unoq_deploy_smoke.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert len(payload["steps"]) == 4
    assert "targeted_pytest" in capsys.readouterr().out


def test_coverage_gate_evaluate_accepts_thresholds() -> None:
    payload = {
        "totals": {"percent_covered": check_air56_unoq_coverage_gate.MIN_TOTAL},
        "files": {
            path: {"summary": {"percent_covered": required}}
            for path, required in check_air56_unoq_coverage_gate.MIN_BY_FILE.items()
        },
    }

    results = check_air56_unoq_coverage_gate._evaluate(payload)
    assert all(row.passed for row in results)


def test_coverage_gate_evaluate_rejects_low_file() -> None:
    payload = {
        "totals": {"percent_covered": check_air56_unoq_coverage_gate.MIN_TOTAL},
        "files": {
            path: {"summary": {"percent_covered": required}}
            for path, required in check_air56_unoq_coverage_gate.MIN_BY_FILE.items()
        },
    }
    payload["files"]["tools/uno_q_protocol.py"]["summary"]["percent_covered"] = 50.0

    results = check_air56_unoq_coverage_gate._evaluate(payload)
    failed = [row.name for row in results if not row.passed]
    assert failed == ["tools/uno_q_protocol.py"]


def test_coverage_gate_reuse_json_fails_cleanly_when_missing(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing_coverage.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_air56_unoq_coverage_gate.py",
            "--reuse-json",
            "--coverage-json",
            str(missing),
        ],
    )

    with pytest.raises(SystemExit, match="coverage JSON not found"):
        check_air56_unoq_coverage_gate.main()
