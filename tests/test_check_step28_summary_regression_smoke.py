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


def _write_baseline(path: Path) -> None:
    payload = {
        "rows": [
            {
                "mode": "mode1",
                "controller": "MIC",
                "metrics": {
                    "avg_power_saving_pct_mean": 0.90,
                    "avg_eta_gain_pct_mean": 0.07,
                    "err_failures_max": 2.0,
                },
                "tolerance": {
                    "abs": {
                        "avg_power_saving_pct_mean": 0.05,
                        "avg_eta_gain_pct_mean": 0.02,
                        "err_failures_max": 0.0,
                    },
                    "rel": {
                        "avg_power_saving_pct_mean": 0.0,
                        "avg_eta_gain_pct_mean": 0.0,
                        "err_failures_max": 0.0,
                    },
                },
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_check_step28_summary_regression_smoke_ok(tmp_path: Path) -> None:
    summary_csv = tmp_path / "step28_ieee_summary.csv"
    baseline_json = tmp_path / "baseline.json"
    out_json = tmp_path / "guard.json"
    out_md = tmp_path / "guard.md"

    _write_csv(
        summary_csv,
        [
            {
                "mode": "mode1",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 0.92,
                "avg_eta_gain_pct_mean": 0.08,
                "err_failures_max": 2.0,
            }
        ],
    )
    _write_baseline(baseline_json)

    cmd = [
        sys.executable,
        "tools/check_step28_summary_regression.py",
        "--summary-csv",
        str(summary_csv),
        "--baseline-json",
        str(baseline_json),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("ok", False)) is True
    assert int(payload.get("checks_failed", 0)) == 0


def test_check_step28_summary_regression_smoke_drift_fail(tmp_path: Path) -> None:
    summary_csv = tmp_path / "step28_ieee_summary.csv"
    baseline_json = tmp_path / "baseline.json"

    _write_csv(
        summary_csv,
        [
            {
                "mode": "mode1",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 0.72,
                "avg_eta_gain_pct_mean": 0.02,
                "err_failures_max": 3.0,
            }
        ],
    )
    _write_baseline(baseline_json)

    cmd = [
        sys.executable,
        "tools/check_step28_summary_regression.py",
        "--summary-csv",
        str(summary_csv),
        "--baseline-json",
        str(baseline_json),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parents[1])
    assert int(proc.returncode) != 0
