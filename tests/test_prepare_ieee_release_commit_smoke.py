from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_prepare_ieee_release_commit_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28" / "tagx"
    ieee = tmp_path / "ieee"
    bundle = ieee / "submission_bundle" / "tagx"

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
    _touch(bundle / "submission_bundle_manifest.md")
    _touch(
        bundle / "submission_bundle_manifest.json",
        '{"bundle_ok": true, "archives":{"zip":"a.zip","tar_gz":"a.tar.gz"}}\n',
    )
    _touch(bundle / "ieee_submission_tagx.zip")
    _touch(bundle / "ieee_submission_tagx.tar.gz")
    _touch(ieee / "FINAL_CHECKLIST_AUTO.md")
    _touch(ieee / "SUBMISSION_CANDIDATE.md")
    _touch(ieee / "SUBMISSION_CANDIDATE.json")
    _touch(ieee / "MANUSCRIPT_CONSISTENCY_REPORT.md")
    _touch(ieee / "MANUSCRIPT_CONSISTENCY_REPORT.json")
    _touch(ieee / "MANUSCRIPT_TEMPLATE_REPORT.md")
    _touch(ieee / "MANUSCRIPT_TEMPLATE_REPORT.json")

    out_json = step28 / "RELEASE_GIT_PLAN.json"
    out_md = step28 / "RELEASE_GIT_PLAN.md"
    cmd = [
        sys.executable,
        "tools/prepare_ieee_release_commit.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee),
        "--tag",
        "tagx",
        "--bundle-dir",
        str(bundle),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("release_ready", False)) is True
    assert int(payload.get("git_add_paths_count", 0)) > 0

