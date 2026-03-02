from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_mode_dir(mode_dir: Path, *, sha: str, pass_mean: bool, pass_worst: bool) -> None:
    final_rows = [
        {
            "controller": "PI",
            "avg_power_saving_pct_mean": 0.0,
            "avg_power_saving_pct_std": 0.0,
            "avg_power_saving_pct_min": 0.0,
            "avg_eta_gain_pct_mean": 0.0,
            "avg_eta_gain_pct_std": 0.0,
            "avg_eta_gain_pct_min": 0.0,
            "err_failures_mean": 0.0,
            "err_failures_max": 0.0,
            "start_stop_power_saving_pct_mean": 0.0,
            "start_stop_power_saving_pct_min": 0.0,
        },
        {
            "controller": "FOC",
            "avg_power_saving_pct_mean": 0.8,
            "avg_power_saving_pct_std": 0.2,
            "avg_power_saving_pct_min": 0.4,
            "avg_eta_gain_pct_mean": 0.2,
            "avg_eta_gain_pct_std": 0.1,
            "avg_eta_gain_pct_min": 0.1,
            "err_failures_mean": 0.0,
            "err_failures_max": 1.0,
            "start_stop_power_saving_pct_mean": 0.2,
            "start_stop_power_saving_pct_min": 0.0,
        },
        {
            "controller": "MIC",
            "avg_power_saving_pct_mean": 1.5,
            "avg_power_saving_pct_std": 0.3,
            "avg_power_saving_pct_min": 0.8,
            "avg_eta_gain_pct_mean": 0.5,
            "avg_eta_gain_pct_std": 0.1,
            "avg_eta_gain_pct_min": 0.2,
            "err_failures_mean": 0.0,
            "err_failures_max": 1.0,
            "start_stop_power_saving_pct_mean": 0.6,
            "start_stop_power_saving_pct_min": 0.2,
        },
    ]
    _write_csv(mode_dir / "step27_final_pi_vs_foc_vs_mic.csv", final_rows)

    _write_csv(
        mode_dir / "step27_stats_motor_controller.csv",
        [
            {
                "motor": "air56",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 1.5,
            }
        ],
    )
    (mode_dir / "step27_air56_acceptance.json").write_text(
        json.dumps({"mean_pass": pass_mean, "worst_case_pass": pass_worst}, ensure_ascii=False),
        encoding="utf-8",
    )
    (mode_dir / "step27_reproducibility.json").write_text(
        json.dumps({"stable_vs_previous": True, "table_sha256": sha}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_step28_summary_smoke(tmp_path: Path) -> None:
    mode1_dir = tmp_path / "mode1"
    mode2_dir = tmp_path / "mode2"
    out_dir = tmp_path / "out"
    _write_mode_dir(mode1_dir, sha="111aaa", pass_mean=True, pass_worst=True)
    _write_mode_dir(mode2_dir, sha="222bbb", pass_mean=False, pass_worst=False)

    cmd = [
        sys.executable,
        "tools/build_step28_ieee_summary.py",
        "--mode1-dir",
        str(mode1_dir),
        "--mode2-dir",
        str(mode2_dir),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    out_csv = out_dir / "step28_ieee_summary.csv"
    out_md = out_dir / "step28_ieee_summary.md"
    assert out_csv.exists()
    assert out_md.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {str(r["mode"]) for r in rows} == {
        "mode1_foc_encoder_vs_mic_sensorless",
        "mode2_foc_sensorless_vs_mic_sensorless",
    }
    assert {str(r["controller"]) for r in rows} == {"PI", "FOC", "MIC"}

    md_text = out_md.read_text(encoding="utf-8")
    assert "Step28 IEEE Summary" in md_text
    assert "mode1_foc_encoder_vs_mic_sensorless" in md_text
    assert "mode2_foc_sensorless_vs_mic_sensorless" in md_text
