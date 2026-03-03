from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_submission_bundle_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28" / "tag1"
    ieee_root = tmp_path / "ieee"

    _touch(step28 / "step28_ieee_summary.csv")
    _touch(step28 / "step28_ieee_summary.md")
    _touch(step28 / "package_manifest.json", "{}\n")
    _touch(step28 / "FINAL_CHECKLIST_AUTO.md")
    _touch(step28 / "submission_candidate_lock.json", '{"lock_ok": true}\n')
    _touch(step28 / "SUBMISSION_CANDIDATE.md")
    _touch(step28 / "SUBMISSION_CANDIDATE.json", '{"ready_for_submission": true}\n')
    _touch(step28 / "RELEASE_COMMIT_MANIFEST.md")
    _touch(step28 / "RELEASE_COMMIT_MANIFEST.json", '{"manifest_ok": true}\n')
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.md")
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.json", '{"status":{"dossier_ok": true}}\n')
    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "MANUSCRIPT_CONSISTENCY_REPORT.md")
    _touch(step28 / "MANUSCRIPT_CONSISTENCY_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "MANUSCRIPT_TEMPLATE_REPORT.md")
    _touch(step28 / "MANUSCRIPT_TEMPLATE_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "derived_ieee" / "ieee_pi_foc_mic_stats.csv")
    _touch(step28 / "derived_ieee" / "motor_tuning_acceptance_summary.csv")
    _touch(step28 / "derived_ieee" / "motor_tuning_acceptance_summary.json", '{"rows":[]}\n')
    _touch(step28 / "passport" / "passport_compare_3motors.csv")
    _touch(step28 / "passport" / "passport_compare_3motors.md")
    _touch(step28 / "passport" / "passport_compare_3motors.json", '{"rows":[]}\n')

    _touch(ieee_root / "manuscript.md")
    _touch(ieee_root / "FINAL_CHECKLIST.md")
    _touch(ieee_root / "guardrails_policy.json", "{}\n")
    _touch(ieee_root / "fig" / "README.md")
    _touch(ieee_root / "fig" / "fig1_mic_methodology.png")
    _touch(ieee_root / "fig" / "fig2_pi_foc_mic_power.pdf")
    _touch(ieee_root / "fig" / "fig3_air56_working_characteristics.pdf")
    _touch(ieee_root / "fig" / "fig4_cross_motor_robustness.pdf")
    _touch(ieee_root / "fig" / "fig5_training_to_foc.pdf")

    out_dir = tmp_path / "bundle_out"
    cmd = [
        sys.executable,
        "tools/build_ieee_submission_bundle.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        "tag1",
        "--out-dir",
        str(out_dir),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    manifest = out_dir / "submission_bundle_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert bool(payload.get("bundle_ok", False)) is True
    archives = dict(payload.get("archives", {}))
    assert Path(str(archives.get("zip", ""))).exists()
    assert Path(str(archives.get("tar_gz", ""))).exists()

