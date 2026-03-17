from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_reproduce_ieee_step28_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "step28_repro"
    pkg_root = tmp_path / "pkg_step28"
    bundle_out = tmp_path / "submission_bundle"
    cmd = [
        sys.executable,
        "tools/reproduce_ieee_step28.py",
        "--out-root",
        str(out_root),
        "--package-root",
        str(pkg_root),
        "--package-tag",
        "smoke01",
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--mic-mode",
        "rule",
        "--skip-air56-tune",
        "--submission-bundle-out-dir",
        str(bundle_out),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert (out_root / "step28_ieee_summary.csv").exists()
    assert (out_root / "step28_ieee_summary.md").exists()
    assert (out_root / "mode1_foc_encoder_vs_mic_sensorless" / "step27_final_pi_vs_foc_vs_mic.csv").exists()
    assert (out_root / "mode2_foc_sensorless_vs_mic_sensorless" / "step27_final_pi_vs_foc_vs_mic.csv").exists()

    pkg_dir = pkg_root / "smoke01"
    assert (pkg_dir / "step28_ieee_summary.csv").exists()
    assert (pkg_dir / "step28_ieee_summary.md").exists()
    assert (pkg_dir / "package_manifest.json").exists()
    assert (pkg_dir / "passport" / "passport_compare_3motors.csv").exists()
    assert (pkg_dir / "passport" / "passport_compare_3motors.md").exists()
    assert (pkg_dir / "passport" / "passport_compare_3motors.json").exists()
    assert (pkg_dir / "derived_ieee" / "ieee_pi_foc_mic_stats.csv").exists()
    assert (pkg_dir / "derived_ieee" / "fig_ieee_pi_foc_mic_power.png").exists()
    assert (pkg_dir / "derived_ieee" / "motor_tuning_acceptance_summary.csv").exists()
    assert (pkg_dir / "derived_ieee" / "motor_air56_tuning_report.md").exists()
    assert not (pkg_dir / "derived_ieee" / "motor_al31_tuning_report.md").exists()
    assert not (pkg_dir / "derived_ieee" / "motor_ao2_tuning_report.md").exists()
    assert (pkg_dir / "submission_candidate_lock.json").exists()
    assert (pkg_dir / "SUBMISSION_CANDIDATE.md").exists()
    assert (pkg_dir / "SUBMISSION_CANDIDATE.json").exists()
    assert (pkg_dir / "RELEASE_COMMIT_MANIFEST.json").exists()
    assert (pkg_dir / "RELEASE_COMMIT_MANIFEST.md").exists()
    assert (pkg_dir / "IEEE_SUBMISSION_DOSSIER.json").exists()
    assert (pkg_dir / "IEEE_SUBMISSION_DOSSIER.md").exists()
    assert (pkg_dir / "FINAL_CHECKLIST_AUTO.md").exists()
    assert (pkg_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json").exists()
    assert (pkg_dir / "MANUSCRIPT_CONSISTENCY_REPORT.md").exists()
    assert (pkg_dir / "MANUSCRIPT_TEMPLATE_REPORT.json").exists()
    assert (pkg_dir / "MANUSCRIPT_TEMPLATE_REPORT.md").exists()
    assert (pkg_dir / "VERIFY_SUBMISSION_CANDIDATE.json").exists()
    assert (bundle_out / "submission_bundle_manifest.json").exists()

    tune_df = pd.read_csv(pkg_dir / "derived_ieee" / "motor_tuning_acceptance_summary.csv")
    assert set(tune_df["motor"].astype(str)) == {"air56"}

    manifest = json.loads((out_root / "step28_reproduce_manifest.json").read_text(encoding="utf-8"))
    assert str(manifest.get("package_tag", "")) == "smoke01"
    assert len(list(manifest.get("executed_commands", []))) >= 12
