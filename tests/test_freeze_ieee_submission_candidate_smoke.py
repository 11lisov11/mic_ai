from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_step28_tree(root: Path) -> None:
    _touch(root / "step28_ieee_summary.csv", "motor,controller\nair56,MIC\nair56,FOC\n")
    _touch(root / "step28_ieee_summary.md")
    _touch(root / "package_manifest.json", "{}\n")

    for mode in MODE_DIRS:
        md = root / mode
        _touch(md / "step27_per_seed_metrics.csv")
        _touch(md / "step27_stats_motor_controller.csv")
        _touch(md / "step27_final_pi_vs_foc_vs_mic.csv")
        _touch(md / "step27_report.md")
        _touch(md / "step27_air56_acceptance.json", '{"mean_pass": true, "worst_case_pass": true}\n')
        _touch(md / "step27_reproducibility.json", '{"table_sha256":"abc","stable_vs_previous":true}\n')

    derived = root / "derived_ieee"
    _touch(derived / "ieee_pi_foc_mic_stats.csv")
    _touch(derived / "ieee_pi_foc_mic_stats.md")
    _touch(derived / "fig_ieee_pi_foc_mic_power.png")
    _touch(derived / "fig_ieee_pi_foc_mic_power.pdf")
    _touch(derived / "fig_ieee_pi_foc_mic_power.svg")
    _touch(derived / "motor_tuning_acceptance_summary.csv")
    _touch(derived / "motor_tuning_acceptance_summary.json", "{}\n")
    _touch(derived / "motor_air56_tuning_report.md")


def test_freeze_submission_candidate_ok(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg"
    _prepare_step28_tree(step28)
    out_json = step28 / "submission_candidate_lock.json"

    cmd = [
        sys.executable,
        "tools/freeze_ieee_submission_candidate.py",
        "--step28-dir",
        str(step28),
        "--out-json",
        str(out_json),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("lock_ok", False)) is True
    assert list(payload.get("required_files_missing", [])) == []
    assert int(payload.get("hashed_files_count", 0)) > 0


def test_freeze_submission_candidate_strict_fails_on_missing_required(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg2"
    _prepare_step28_tree(step28)
    (step28 / "mode1_foc_encoder_vs_mic_sensorless" / "step27_report.md").unlink()

    cmd = [
        sys.executable,
        "tools/freeze_ieee_submission_candidate.py",
        "--step28-dir",
        str(step28),
        "--strict",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

