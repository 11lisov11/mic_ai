from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_submission_dossier_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg"
    ieee = tmp_path / "ieee"
    (step28 / "derived_ieee").mkdir(parents=True, exist_ok=True)
    (step28 / "passport").mkdir(parents=True, exist_ok=True)
    ieee.mkdir(parents=True, exist_ok=True)

    _touch(step28 / "FINAL_CHECKLIST_AUTO.md", "- [x] ready_for_submission: `True`\n")
    _touch(step28 / "submission_candidate_lock.json", '{"lock_ok": true, "aggregate_sha256": "a"}\n')
    _touch(step28 / "SUBMISSION_CANDIDATE.json", '{"ready_for_submission": true}\n')
    _touch(step28 / "RELEASE_COMMIT_MANIFEST.json", '{"manifest_ok": true, "aggregate_sha256": "b", "git": {"commit": "c", "branch": "main", "dirty": false}}\n')
    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "MANUSCRIPT_TEMPLATE_REPORT.json", '{"ok": true}\n')
    _touch(
        step28 / "derived_ieee" / "motor_tuning_acceptance_summary.json",
        '{"rows":[{"motor":"air56","acceptance_pass":true,"avg_power_saving_pct_mean":0.6,"avg_power_saving_pct_min":0.6,"avg_eta_gain_pct_mean":0.1,"avg_eta_gain_pct_min":0.1,"err_failures_max":0}]}\n',
    )
    _touch(step28 / "passport" / "passport_compare_3motors.json", '{"rows":[],"warnings":[],"failures":[]}\n')

    out_json = step28 / "IEEE_SUBMISSION_DOSSIER.json"
    out_md = step28 / "IEEE_SUBMISSION_DOSSIER.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_submission_dossier.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    status = dict(payload.get("status", {}))
    assert bool(status.get("dossier_ok", False)) is True
