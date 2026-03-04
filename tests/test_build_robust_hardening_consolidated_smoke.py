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


def _write_run(base: Path, run_name: str, motor: str, selected_tag: str) -> Path:
    run_dir = base / run_name
    motor_dir = run_dir / motor
    motor_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "results": [
            {
                "motor": motor,
                "selection_policy": "safe_baseline_guard",
                "improved_vs_baseline": True,
                "config_applied": False,
                "thresholds": {
                    "baseline_min_power": 0.2,
                    "baseline_min_eta": 0.0,
                    "baseline_max_err": 2.0,
                    "baseline_min_start_stop": -0.5,
                },
                "selected_candidate": {
                    "tag": selected_tag,
                    "robust_score": 1.5,
                    "baseline_power": 0.8,
                    "perturb_power_min": 0.2,
                    "perturb_eta_min": 0.01,
                    "perturb_err_max": 1.0,
                    "perturb_start_stop_min": -0.1,
                    "robust_pass": True,
                },
            }
        ]
    }
    (run_dir / "robust_hardening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        motor_dir / f"{motor}_robust_stage2_rank.csv",
        [
            {
                "tag": "baseline",
                "source": "config",
                "robust_score": 5.0,
                "robust_pass": False,
                "baseline_power": 0.3,
                "baseline_eta": 0.0,
                "baseline_err": 1.0,
                "baseline_start_stop": -0.2,
                "perturb_power_min": -0.1,
                "perturb_eta_min": -0.2,
            },
            {
                "tag": selected_tag,
                "source": "random_local_safe",
                "robust_score": 1.5,
                "robust_pass": True,
                "baseline_power": 0.8,
                "baseline_eta": 0.02,
                "baseline_err": 1.0,
                "baseline_start_stop": -0.1,
                "perturb_power_min": 0.2,
                "perturb_eta_min": 0.01,
            },
        ],
    )
    return run_dir


def test_build_robust_hardening_consolidated_smoke(tmp_path: Path) -> None:
    run1 = _write_run(tmp_path, "robust_hardening_a", "ao2", "cand_a")
    run2 = _write_run(tmp_path, "robust_hardening_b", "al31", "cand_b")
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "tools/build_robust_hardening_consolidated.py",
        "--runs",
        f"{run1},{run2}",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    rank_csv = out_dir / "robust_hardening_consolidated_rank.csv"
    selected_csv = out_dir / "robust_hardening_consolidated_selected.csv"
    best_csv = out_dir / "robust_hardening_consolidated_best.csv"
    report_json = out_dir / "robust_hardening_consolidated.json"
    report_md = out_dir / "robust_hardening_consolidated.md"
    assert rank_csv.exists()
    assert selected_csv.exists()
    assert best_csv.exists()
    assert report_json.exists()
    assert report_md.exists()

    with rank_csv.open("r", encoding="utf-8", newline="") as handle:
        rank_rows = list(csv.DictReader(handle))
    assert len(rank_rows) == 4

    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        best_rows = list(csv.DictReader(handle))
    assert {str(r["motor"]) for r in best_rows} == {"ao2", "al31"}

