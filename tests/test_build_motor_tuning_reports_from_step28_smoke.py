from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_mode_stats(path: Path, *, air56_power: float, al31_power: float, ao2_power: float) -> None:
    rows = [
        {
            "motor": "air56",
            "controller": "MIC",
            "samples": 5,
            "avg_power_saving_pct_mean": air56_power,
            "avg_power_saving_pct_min": air56_power,
            "avg_eta_gain_pct_mean": 0.1,
            "avg_eta_gain_pct_min": 0.1,
            "err_failures_mean": 1.0,
            "err_failures_max": 2.0,
            "start_stop_power_saving_pct_mean": -0.2,
            "start_stop_power_saving_pct_min": -0.2,
        },
        {
            "motor": "al31",
            "controller": "MIC",
            "samples": 5,
            "avg_power_saving_pct_mean": al31_power,
            "avg_power_saving_pct_min": al31_power,
            "avg_eta_gain_pct_mean": 0.05,
            "avg_eta_gain_pct_min": 0.05,
            "err_failures_mean": 0.0,
            "err_failures_max": 0.0,
            "start_stop_power_saving_pct_mean": 1.5,
            "start_stop_power_saving_pct_min": 1.5,
        },
        {
            "motor": "ao2",
            "controller": "MIC",
            "samples": 5,
            "avg_power_saving_pct_mean": ao2_power,
            "avg_power_saving_pct_min": ao2_power,
            "avg_eta_gain_pct_mean": 0.01,
            "avg_eta_gain_pct_min": 0.01,
            "err_failures_mean": 0.0,
            "err_failures_max": 0.0,
            "start_stop_power_saving_pct_mean": -0.01,
            "start_stop_power_saving_pct_min": -0.01,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_motor_tuning_reports_from_step28_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28_pkg"
    m1 = step28 / "mode1_foc_encoder_vs_mic_sensorless"
    m2 = step28 / "mode2_foc_sensorless_vs_mic_sensorless"
    m1.mkdir(parents=True, exist_ok=True)
    m2.mkdir(parents=True, exist_ok=True)
    _write_mode_stats(m1 / "step27_stats_motor_controller.csv", air56_power=0.6, al31_power=1.0, ao2_power=0.1)
    _write_mode_stats(m2 / "step27_stats_motor_controller.csv", air56_power=0.7, al31_power=0.8, ao2_power=0.05)

    out_dir = tmp_path / "derived"
    cmd = [
        sys.executable,
        "tools/build_motor_tuning_reports_from_step28.py",
        "--step28-dir",
        str(step28),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert (out_dir / "motor_air56_tuning_report.md").exists()
    assert (out_dir / "motor_al31_tuning_report.md").exists()
    assert (out_dir / "motor_ao2_tuning_report.md").exists()
    assert (out_dir / "motor_air56_search_rank.csv").exists()
    assert (out_dir / "motor_al31_search_rank.csv").exists()
    assert (out_dir / "motor_ao2_search_rank.csv").exists()
    summary_json = out_dir / "motor_tuning_acceptance_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    assert len(rows) == 3
    air56 = [r for r in rows if str(r.get("motor")) == "air56"][0]
    assert bool(air56.get("acceptance_pass", False)) is True
