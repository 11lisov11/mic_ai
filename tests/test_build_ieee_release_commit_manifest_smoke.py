from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_release_commit_manifest_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "pkg"
    ieee_root = tmp_path / "ieee"
    tag = "smoke_tag"
    (step28 / "derived_ieee").mkdir(parents=True, exist_ok=True)
    (step28 / "passport").mkdir(parents=True, exist_ok=True)
    (ieee_root / "data" / "release" / tag).mkdir(parents=True, exist_ok=True)

    _touch(step28 / "step28_ieee_summary.csv")
    _touch(step28 / "step28_ieee_summary.md")
    _touch(step28 / "package_manifest.json", "{}\n")
    _touch(step28 / "FINAL_CHECKLIST_AUTO.md")
    _touch(step28 / "submission_candidate_lock.json", '{"lock_ok": true, "required_files_missing": []}\n')
    _touch(step28 / "SUBMISSION_CANDIDATE.md")
    _touch(step28 / "SUBMISSION_CANDIDATE.json", "{}\n")
    _touch(step28 / "derived_ieee" / "ieee_pi_foc_mic_stats.csv")
    _touch(step28 / "derived_ieee" / "motor_tuning_acceptance_summary.json", '{"rows":[]}\n')
    _touch(step28 / "passport" / "passport_compare_3motors.json", '{"rows":[],"warnings":[],"failures":[]}\n')
    _touch(ieee_root / "guardrails_policy.json", '{"motor_saving_thresholds_pct":{"air56":0.5,"al31":0.0,"ao2":0.05}}\n')
    _touch(ieee_root / "manuscript.md")
    _touch(ieee_root / "FINAL_CHECKLIST_AUTO.md")
    _touch(ieee_root / "SUBMISSION_CANDIDATE.md")
    _touch(ieee_root / "SUBMISSION_CANDIDATE.json", "{}\n")
    _touch(ieee_root / "data" / "release" / tag / "promotion_manifest.json", "{}\n")
    _touch(ieee_root / "data" / "release" / tag / "release_snapshot.json", "{}\n")

    out_json = step28 / "RELEASE_COMMIT_MANIFEST.json"
    out_md = step28 / "RELEASE_COMMIT_MANIFEST.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_release_commit_manifest.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        tag,
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--allow-dirty",
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("required_ok", False)) is True
    assert bool(payload.get("manifest_ok", False)) is True
    assert int(payload.get("files_count", 0)) > 0
    assert str(payload.get("aggregate_sha256", ""))
