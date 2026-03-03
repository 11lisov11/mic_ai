from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_ieee_step28_strict(tmp_path: Path) -> None:
    src = tmp_path / "step28_out"
    src.mkdir(parents=True, exist_ok=True)
    (src / "step28_ieee_summary.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (src / "step28_ieee_summary.md").write_text("# ok\n", encoding="utf-8")

    mode_files = (
        "step27_per_seed_metrics.csv",
        "step27_stats_motor_controller.csv",
        "step27_final_pi_vs_foc_vs_mic.csv",
        "step27_air56_acceptance.json",
        "step27_reproducibility.json",
        "step27_report.md",
        "step27_seed_perturbations.csv",
    )
    for mode in ("mode1_foc_encoder_vs_mic_sensorless", "mode2_foc_sensorless_vs_mic_sensorless"):
        mode_dir = src / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for name in mode_files:
            (mode_dir / name).write_text("x\n", encoding="utf-8")

    out_root = tmp_path / "pkg_out"
    cmd = [
        sys.executable,
        "scripts/package_ieee_step28.py",
        "--step28-out",
        str(src),
        "--dest-root",
        str(out_root),
        "--tag",
        "t01",
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    pkg = out_root / "t01"
    assert (pkg / "step28_ieee_summary.csv").exists()
    assert (pkg / "step28_ieee_summary.md").exists()
    assert (pkg / "mode1_foc_encoder_vs_mic_sensorless" / "step27_final_pi_vs_foc_vs_mic.csv").exists()
    assert (pkg / "mode2_foc_sensorless_vs_mic_sensorless" / "step27_final_pi_vs_foc_vs_mic.csv").exists()
    assert (pkg / "package_manifest.json").exists()
