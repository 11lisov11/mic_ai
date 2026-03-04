from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_rebuttal_evidence_pack_smoke(tmp_path: Path) -> None:
    tag = "tagx"
    step28 = tmp_path / "step28" / tag
    ieee_root = tmp_path / "ieee"
    bundle = ieee_root / "submission_bundle" / tag
    mode1 = step28 / "mode1_foc_encoder_vs_mic_sensorless"
    mode2 = step28 / "mode2_foc_sensorless_vs_mic_sensorless"

    _touch(step28 / "step28_ieee_summary.csv", "mode,controller\nm,MIC\n")
    _touch(step28 / "step28_ieee_summary.md", "# summary\n")
    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "FINAL_CHECKLIST_AUTO.md", "- [x] ready_for_submission: `True`\n")
    _touch(step28 / "MANUSCRIPT_CONSISTENCY_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "MANUSCRIPT_TEMPLATE_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.json", '{"status":{"dossier_ok": true}}\n')
    _touch(step28 / "IEEE_SUBMISSION_HANDOFF.json", '{"handoff_ready": true}\n')
    _touch(step28 / "RELEASE_COMMIT_MANIFEST.json", '{"manifest_ok": true}\n')
    _touch(step28 / "IEEE_RELEASE_NOTES.json", '{"strict_ready": true}\n')
    _touch(step28 / "STEP28_REGRESSION_GUARD.json", '{"ok": true}\n')
    _touch(step28 / "CAMERA_READY_CHECKLIST.json", '{"camera_ready_ok": true}\n')

    _touch(mode1 / "step27_per_seed_metrics.csv", "motor,seed\nair56,101\n")
    _touch(mode1 / "step27_stats_motor_controller.csv", "motor,controller\nair56,MIC\n")
    _touch(mode1 / "step27_final_pi_vs_foc_vs_mic.csv", "controller\nMIC\n")
    _touch(mode2 / "step27_per_seed_metrics.csv", "motor,seed\nair56,101\n")
    _touch(mode2 / "step27_stats_motor_controller.csv", "motor,controller\nair56,MIC\n")
    _touch(mode2 / "step27_final_pi_vs_foc_vs_mic.csv", "controller\nMIC\n")

    _touch(step28 / "derived_ieee" / "ieee_pi_foc_mic_stats.csv", "x\n")
    _touch(step28 / "derived_ieee" / "fig_ieee_pi_foc_mic_power.pdf", "x\n")
    _touch(step28 / "passport" / "passport_compare_3motors.csv", "x\n")
    _touch(step28 / "passport" / "passport_compare_3motors.json", '{"rows":[],"warnings":[],"failures":[]}\n')

    _touch(bundle / "submission_bundle_manifest.json", '{"bundle_ok": true}\n')
    _touch(ieee_root / "manuscript.md", "# m\n")
    _touch(ieee_root / "guardrails_policy.json", '{"motor_saving_thresholds_pct":{"air56":0.5}}\n')

    out_dir = tmp_path / "rebuttal"
    cmd = [
        sys.executable,
        "tools/build_ieee_rebuttal_evidence_pack.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        tag,
        "--out-dir",
        str(out_dir),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    manifest = out_dir / "REBUTTAL_EVIDENCE_PACK.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert bool(payload.get("strict_ready", False)) is True
    assert int(payload.get("copied_files_count", 0)) > 0
