from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_per_seed_csv(path: Path, mode_shift: float) -> None:
    rows = []
    for motor in ("air56", "al31", "ao2"):
        for seed in (101, 202):
            rows.extend(
                [
                    {
                        "motor": motor,
                        "seed": seed,
                        "controller": "PI",
                        "avg_power_saving_pct": 0.0,
                        "avg_eta_gain_pct": 0.0,
                        "err_failures": 0.0,
                        "worst_current_peak_ratio": 1.0,
                        "worst_current_mean_ratio": 1.0,
                        "avg_controller_speed_err": 1.0,
                    },
                    {
                        "motor": motor,
                        "seed": seed,
                        "controller": "FOC",
                        "avg_power_saving_pct": 0.0,
                        "avg_eta_gain_pct": 0.0,
                        "err_failures": 0.0,
                        "worst_current_peak_ratio": 1.0,
                        "worst_current_mean_ratio": 1.0,
                        "avg_controller_speed_err": 1.0,
                    },
                    {
                        "motor": motor,
                        "seed": seed,
                        "controller": "MIC",
                        "avg_power_saving_pct": 2.0 + mode_shift,
                        "avg_eta_gain_pct": 1.0 + mode_shift,
                        "err_failures": 0.0,
                        "worst_current_peak_ratio": 1.0,
                        "worst_current_mean_ratio": 1.0,
                        "avg_controller_speed_err": 1.0,
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_ieee_figures_tables_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28_pkg"
    m1 = step28 / "mode1_foc_encoder_vs_mic_sensorless"
    m2 = step28 / "mode2_foc_sensorless_vs_mic_sensorless"
    m1.mkdir(parents=True, exist_ok=True)
    m2.mkdir(parents=True, exist_ok=True)
    _write_per_seed_csv(m1 / "step27_per_seed_metrics.csv", mode_shift=0.0)
    _write_per_seed_csv(m2 / "step27_per_seed_metrics.csv", mode_shift=0.5)

    out_dir = tmp_path / "derived"
    cmd = [
        sys.executable,
        "tools/build_ieee_figures_tables.py",
        "--step28-dir",
        str(step28),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    out_csv = out_dir / "ieee_pi_foc_mic_stats.csv"
    out_md = out_dir / "ieee_pi_foc_mic_stats.md"
    out_png = out_dir / "fig_ieee_pi_foc_mic_power.png"
    out_pdf = out_dir / "fig_ieee_pi_foc_mic_power.pdf"
    out_svg = out_dir / "fig_ieee_pi_foc_mic_power.svg"
    assert out_csv.exists()
    assert out_md.exists()
    assert out_png.exists()
    assert out_pdf.exists()
    assert out_svg.exists()

    df = pd.read_csv(out_csv)
    assert set(df["mode"].astype(str)) == {"mode1", "mode2"}
    assert set(df["controller"].astype(str)) == {"PI", "FOC", "MIC"}
