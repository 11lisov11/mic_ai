from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_promote_ieee_release_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    step28_dir = tmp_path / "step28_pkg"
    derived = step28_dir / "derived_ieee"
    passport = step28_dir / "passport"
    ieee_root = tmp_path / "paper_ieee"
    pgups_fig = tmp_path / "pgups_fig"

    # Minimal frozen summary with MIC rows.
    df = pd.DataFrame(
        [
            {
                "mode": "mode1_foc_encoder_vs_mic_sensorless",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 1.0,
                "avg_power_saving_pct_min": 0.1,
                "avg_eta_gain_pct_mean": 0.05,
                "avg_eta_gain_pct_min": 0.01,
                "err_failures_max": 2.0,
                "start_stop_power_saving_pct_mean": 2.5,
                "table_sha256": "abc",
            },
            {
                "mode": "mode2_foc_sensorless_vs_mic_sensorless",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 1.1,
                "avg_power_saving_pct_min": 0.2,
                "avg_eta_gain_pct_mean": 0.06,
                "avg_eta_gain_pct_min": 0.02,
                "err_failures_max": 2.0,
                "start_stop_power_saving_pct_mean": 2.7,
                "table_sha256": "abc",
            },
        ]
    )
    step28_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(step28_dir / "step28_ieee_summary.csv", index=False)
    _touch(step28_dir / "step28_ieee_summary.md", "# summary")
    _touch(step28_dir / "package_manifest.json", json.dumps({"ok": True}))

    for ext in (".png", ".pdf", ".svg"):
        _touch(derived / f"fig_ieee_pi_foc_mic_power{ext}", "bin")
    _touch(derived / "ieee_pi_foc_mic_stats.csv", "a,b\n1,2\n")
    _touch(derived / "ieee_pi_foc_mic_stats.md", "# stats")
    _touch(derived / "motor_tuning_acceptance_summary.csv", "motor,pass\nair56,True\n")
    _touch(derived / "motor_tuning_acceptance_summary.json", '{"ok": true}')
    _touch(derived / "motor_air56_tuning_report.md", "# air56")

    _touch(passport / "passport_compare_3motors.csv", "motor,delta\nair56,0.1\n")

    # Optional PGUPS figure sources.
    _touch(pgups_fig / "working_characteristics_air56_foc_mic.png", "img")
    _touch(pgups_fig / "fig_multi_motor_scenario_heatmap_ru.png", "img")
    _touch(pgups_fig / "fig_learning_vs_foc_ru.png", "img")
    _touch(pgups_fig / "fig_algorithm_block_ru.png", "img")

    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tools/promote_ieee_release.py",
        "--step28-dir",
        str(step28_dir),
        "--ieee-root",
        str(ieee_root),
        "--pgups-fig-dir",
        str(pgups_fig),
        "--tag",
        "smoke_release",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    assert (ieee_root / "fig" / "fig2_pi_foc_mic_power.png").exists()
    assert (ieee_root / "fig" / "fig2_pi_foc_mic_power.pdf").exists()
    assert (ieee_root / "fig" / "fig2_pi_foc_mic_power.svg").exists()
    assert (ieee_root / "fig" / "fig3_air56_working_characteristics.png").exists()
    assert (ieee_root / "fig" / "fig4_cross_motor_robustness.png").exists()
    assert (ieee_root / "fig" / "fig5_training_to_foc.png").exists()
    assert (ieee_root / "fig" / "fig1_mic_methodology.png").exists()

    release = ieee_root / "data" / "release" / "smoke_release"
    assert (release / "tables" / "step28_ieee_summary.csv").exists()
    assert (release / "tables" / "ieee_pi_foc_mic_stats.csv").exists()
    assert (release / "release_snapshot.json").exists()
    assert (release / "promotion_manifest.json").exists()

    snap = json.loads((release / "release_snapshot.json").read_text(encoding="utf-8"))
    assert snap["mic_rows"] == 2
    assert float(snap["avg_power_saving_pct_mean"]) > 0.0
