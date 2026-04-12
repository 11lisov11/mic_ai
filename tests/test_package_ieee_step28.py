from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _prepare_minimal_step28_tree(src: Path) -> None:
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


def test_package_ieee_step28_strict(tmp_path: Path) -> None:
    src = tmp_path / "step28_out"
    _prepare_minimal_step28_tree(src)

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
    assert (pkg / "mode1_foc_encoder_vs_mic_sensorless" / "step27_motor_acceptance.json").exists()
    assert (pkg / "mode1_foc_encoder_vs_mic_sensorless" / "step27_air56_acceptance.json").exists()
    assert (pkg / "mode2_foc_sensorless_vs_mic_sensorless" / "step27_final_pi_vs_foc_vs_mic.csv").exists()
    assert (pkg / "mode2_foc_sensorless_vs_mic_sensorless" / "step27_motor_acceptance.json").exists()
    assert (pkg / "mode2_foc_sensorless_vs_mic_sensorless" / "step27_air56_acceptance.json").exists()
    assert (pkg / "package_manifest.json").exists()


def test_package_ieee_step28_with_theory_and_passport(tmp_path: Path) -> None:
    src = tmp_path / "step28_out"
    _prepare_minimal_step28_tree(src)

    theory_csv = tmp_path / "working_chars.csv"
    theory_csv.write_text(
        "\n".join(
            [
                "policy,p2_kw,m2,i_rms,n2_rpm,eta_pct,cos_phi,p_el_pos",
                "FOC,0.05,0.3,0.22,1384,62.0,0.32,80.6",
                "FOC,0.10,0.6,0.27,1383,70.0,0.52,142.9",
                "FOC,0.15,0.9,0.33,1382,75.0,0.68,200.0",
                "FOC,0.20,1.2,0.41,1381,77.0,0.79,259.7",
            ]
        ),
        encoding="utf-8",
    )

    passport_src = tmp_path / "passport_src"
    passport_src.mkdir(parents=True, exist_ok=True)
    (passport_src / "passport_compare_3motors.csv").write_text("motor,policy\nair56,FOC\n", encoding="utf-8")
    (passport_src / "passport_compare_3motors.md").write_text("# passport\n", encoding="utf-8")
    (passport_src / "passport_compare_3motors.json").write_text("{\"rows\": [], \"failures\": []}\n", encoding="utf-8")

    out_root = tmp_path / "pkg_out"
    cmd = [
        sys.executable,
        "scripts/package_ieee_step28.py",
        "--step28-out",
        str(src),
        "--dest-root",
        str(out_root),
        "--tag",
        "t02",
        "--strict",
        "--theory-csv",
        str(theory_csv),
        "--passport-dir",
        str(passport_src),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    pkg = out_root / "t02"
    assert (pkg / "theory_validation" / "report.json").exists()
    assert (pkg / "theory_validation" / "report.md").exists()
    assert (pkg / "passport" / "passport_compare_3motors.csv").exists()
    assert (pkg / "passport" / "passport_compare_3motors.md").exists()
    assert (pkg / "passport" / "passport_compare_3motors.json").exists()
