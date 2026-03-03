from __future__ import annotations

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


def _prepare_ready_tree(root: Path) -> None:
    _touch(root / "step28_ieee_summary.csv")
    _touch(root / "step28_ieee_summary.md")
    _touch(root / "package_manifest.json", "{}\n")

    for mode in MODE_DIRS:
        md = root / mode
        _touch(md / "step27_per_seed_metrics.csv")
        _touch(md / "step27_stats_motor_controller.csv")
        _touch(md / "step27_final_pi_vs_foc_vs_mic.csv")
        _touch(md / "step27_report.md")
        _touch(md / "step27_air56_acceptance.json", '{"mean_pass": true, "worst_case_pass": true}\n')
        _touch(md / "step27_reproducibility.json", '{"table_sha256":"abc","stable_vs_previous":true}\n')

    derived = root / "derived_ieee"
    _touch(derived / "ieee_pi_foc_mic_stats.csv")
    _touch(derived / "ieee_pi_foc_mic_stats.md")
    _touch(derived / "fig_ieee_pi_foc_mic_power.png")
    _touch(derived / "fig_ieee_pi_foc_mic_power.pdf")
    _touch(derived / "fig_ieee_pi_foc_mic_power.svg")
    _touch(
        derived / "motor_tuning_acceptance_summary.json",
        (
            '{"rows": ['
            '{"motor":"air56","avg_power_saving_pct_mean":0.8,"avg_power_saving_pct_min":0.7,"acceptance_pass":true},'
            '{"motor":"al31","avg_power_saving_pct_mean":0.2,"avg_power_saving_pct_min":0.1,"acceptance_pass":true},'
            '{"motor":"ao2","avg_power_saving_pct_mean":0.08,"avg_power_saving_pct_min":0.06,"acceptance_pass":true}'
            "]}\n"
        ),
    )

    passport = root / "passport"
    _touch(passport / "passport_compare_3motors.csv", "motor,policy\nair56,FOC\n")
    _touch(passport / "passport_compare_3motors.md", "# passport\n")
    _touch(passport / "passport_compare_3motors.json", '{"rows": [], "warnings": [], "failures": []}\n')


def test_build_ieee_final_checklist_ready_strict(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    out_md = tmp_path / "checklist.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `True`" in text


def test_build_ieee_final_checklist_detects_blocker(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg2"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee2"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    (step28 / MODE_DIRS[0] / "step27_air56_acceptance.json").write_text(
        '{"mean_pass": false, "worst_case_pass": true}\n', encoding="utf-8"
    )
    out_md = tmp_path / "checklist2.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `False`" in text
    assert "AIR56 acceptance gate is not fully satisfied" in text


def test_build_ieee_final_checklist_detects_passport_failure(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg3"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee3"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    (step28 / "passport" / "passport_compare_3motors.json").write_text(
        '{"rows": [], "warnings": [], "failures": [{"motor":"ao2","error":"bad"}]}\n', encoding="utf-8"
    )

    out_md = tmp_path / "checklist3.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `False`" in text
    assert "passport failures are present" in text


def test_build_ieee_final_checklist_require_lock(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg4"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee4"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    (step28 / "submission_candidate_lock.json").write_text(
        '{"lock_ok": true, "required_files_missing": []}\n', encoding="utf-8"
    )

    out_md = tmp_path / "checklist4.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
        "--require-lock",
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `True`" in text


def test_build_ieee_final_checklist_detects_motor_guardrail_failure(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg5"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee5"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    (step28 / "submission_candidate_lock.json").write_text(
        '{"lock_ok": true, "required_files_missing": []}\n', encoding="utf-8"
    )
    # Force AO2 below default threshold (0.05%).
    (step28 / "derived_ieee" / "motor_tuning_acceptance_summary.json").write_text(
        (
            '{"rows": ['
            '{"motor":"air56","avg_power_saving_pct_mean":0.8,"avg_power_saving_pct_min":0.7,"acceptance_pass":true},'
            '{"motor":"al31","avg_power_saving_pct_mean":0.2,"avg_power_saving_pct_min":0.1,"acceptance_pass":true},'
            '{"motor":"ao2","avg_power_saving_pct_mean":0.01,"avg_power_saving_pct_min":0.01,"acceptance_pass":true}'
            "]}\n"
        ),
        encoding="utf-8",
    )

    out_md = tmp_path / "checklist5.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
        "--require-lock",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `False`" in text
    assert "motor acceptance guardrails failed" in text


def test_build_ieee_final_checklist_policy_override_failure(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg6"
    _prepare_ready_tree(step28)
    ieee_root = tmp_path / "ieee6"
    (ieee_root / "fig").mkdir(parents=True, exist_ok=True)
    (ieee_root / "manuscript.md").write_text("# m\n", encoding="utf-8")
    for rel in (
        "fig1_mic_methodology.png",
        "fig2_pi_foc_mic_power.pdf",
        "fig3_air56_working_characteristics.pdf",
        "fig4_cross_motor_robustness.pdf",
        "fig5_training_to_foc.pdf",
    ):
        (ieee_root / "fig" / rel).write_text("x\n", encoding="utf-8")
    (step28 / "submission_candidate_lock.json").write_text(
        '{"lock_ok": true, "required_files_missing": []}\n', encoding="utf-8"
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        '{"motor_saving_thresholds_pct":{"air56":0.5,"al31":0.0,"ao2":0.2}}\n',
        encoding="utf-8",
    )

    out_md = tmp_path / "checklist6.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28),
        "--out-md",
        str(out_md),
        "--ieee-root",
        str(ieee_root),
        "--guardrails-policy",
        str(policy),
        "--require-lock",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    text = out_md.read_text(encoding="utf-8")
    assert "ready_for_submission: `False`" in text
    assert "motor acceptance guardrails failed" in text
