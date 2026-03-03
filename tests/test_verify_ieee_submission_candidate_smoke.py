from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_step28_and_ieee(step28: Path, ieee_root: Path) -> None:
    _touch(step28 / "step28_ieee_summary.csv", "mode,controller,avg_power_saving_pct_mean,avg_power_saving_pct_min,avg_eta_gain_pct_mean,avg_eta_gain_pct_min,err_failures_max\na,MIC,0.6,0.6,0.1,0.1,0\n")
    _touch(step28 / "step28_ieee_summary.md")
    _touch(step28 / "package_manifest.json", "{}\n")
    _touch(step28 / "submission_candidate_lock.json", '{"lock_ok": true, "required_files_missing": []}\n')

    for mode in MODE_DIRS:
        md = step28 / mode
        _touch(md / "step27_per_seed_metrics.csv")
        _touch(md / "step27_stats_motor_controller.csv")
        _touch(md / "step27_final_pi_vs_foc_vs_mic.csv")
        _touch(md / "step27_report.md")
        _touch(md / "step27_air56_acceptance.json", '{"mean_pass": true, "worst_case_pass": true}\n')
        _touch(md / "step27_reproducibility.json", '{"table_sha256":"abc","stable_vs_previous":true}\n')

    derived = step28 / "derived_ieee"
    _touch(derived / "ieee_pi_foc_mic_stats.csv")
    _touch(derived / "ieee_pi_foc_mic_stats.md")
    _touch(derived / "fig_ieee_pi_foc_mic_power.png")
    _touch(derived / "fig_ieee_pi_foc_mic_power.pdf")
    _touch(derived / "fig_ieee_pi_foc_mic_power.svg")
    _touch(
        derived / "motor_tuning_acceptance_summary.json",
        (
            '{"rows": ['
            '{"motor":"air56","avg_power_saving_pct_mean":0.6,"avg_power_saving_pct_min":0.6,"acceptance_pass":true},'
            '{"motor":"al31","avg_power_saving_pct_mean":0.1,"avg_power_saving_pct_min":0.1,"acceptance_pass":true},'
            '{"motor":"ao2","avg_power_saving_pct_mean":0.06,"avg_power_saving_pct_min":0.06,"acceptance_pass":true}'
            "]}\n"
        ),
    )

    passport = step28 / "passport"
    _touch(passport / "passport_compare_3motors.csv")
    _touch(passport / "passport_compare_3motors.md")
    _touch(passport / "passport_compare_3motors.json", '{"rows":[],"warnings":[],"failures":[]}\n')

    _touch(
        ieee_root / "manuscript.md",
        (
            "## Abstract\n"
            "This abstract contains enough words to satisfy the template checker for a smoke test, "
            "while keeping the content synthetic and deterministic for CI validation.\n"
            "## I. Introduction\n"
            "See Fig. 2 in `fig/fig2_pi_foc_mic_power.pdf`.\n"
            "## II. Method\n"
            "Method details.\n"
            "## III. Experimental Setup\n"
            "Setup details.\n"
            "## IV. Results\n"
            "Main results are summarized in Table 1 from `../pkg/step28_ieee_summary.csv`.\n"
            "## V. Theory Validation\n"
            "Validation details.\n"
            "## VI. Discussion\n"
            "Discussion details.\n"
            "## VII. Conclusion\n"
            "Conclusion details.\n"
        ),
    )
    _touch(ieee_root / "fig" / "fig1_mic_methodology.png")
    _touch(ieee_root / "fig" / "fig2_pi_foc_mic_power.pdf")
    _touch(ieee_root / "fig" / "fig3_air56_working_characteristics.pdf")
    _touch(ieee_root / "fig" / "fig4_cross_motor_robustness.pdf")
    _touch(ieee_root / "fig" / "fig5_training_to_foc.pdf")


def test_verify_ieee_submission_candidate_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg"
    ieee_root = tmp_path / "ieee"
    _prepare_step28_and_ieee(step28, ieee_root)

    policy = tmp_path / "guardrails_policy.json"
    policy.write_text(
        '{"motor_saving_thresholds_pct":{"air56":0.5,"al31":0.0,"ao2":0.05}}\n',
        encoding="utf-8",
    )
    out_json = step28 / "VERIFY_SUBMISSION_CANDIDATE.json"

    cmd = [
        sys.executable,
        "tools/verify_ieee_submission_candidate.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee_root),
        "--manuscript",
        str(ieee_root / "manuscript.md"),
        "--guardrails-policy",
        str(policy),
        "--allow-dirty",
        "--out-json",
        str(out_json),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("verification_ok", False)) is True
    assert bool(payload.get("checklist_ready_for_submission", False)) is True
    assert bool(payload.get("release_manifest_ok", False)) is True
    assert bool(payload.get("manuscript_template_ok", False)) is True
    artifacts = dict(payload.get("artifacts", {}))
    assert Path(str(artifacts.get("dossier_json", ""))).exists()
    assert Path(str(artifacts.get("dossier_md", ""))).exists()
    assert Path(str(artifacts.get("manuscript_consistency_json", ""))).exists()
    assert Path(str(artifacts.get("manuscript_consistency_md", ""))).exists()
    assert Path(str(artifacts.get("manuscript_template_json", ""))).exists()
    assert Path(str(artifacts.get("manuscript_template_md", ""))).exists()
