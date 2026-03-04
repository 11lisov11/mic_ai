from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_camera_ready_checklist_smoke(tmp_path: Path) -> None:
    tag = "tagx"
    step28 = tmp_path / "step28" / tag
    ieee_root = tmp_path / "ieee"
    bundle = ieee_root / "submission_bundle" / tag

    _touch(step28 / "FINAL_CHECKLIST_AUTO.md", "- [x] ready_for_submission: `True`\n")
    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "MANUSCRIPT_TEMPLATE_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "MANUSCRIPT_CONSISTENCY_REPORT.json", '{"ok": true}\n')
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.json", '{"status":{"dossier_ok": true}}\n')
    _touch(step28 / "IEEE_SUBMISSION_HANDOFF.json", '{"handoff_ready": true}\n')
    _touch(step28 / "IEEE_RELEASE_NOTES.json", '{"strict_ready": true}\n')
    _touch(step28 / "STEP28_REGRESSION_GUARD.json", '{"ok": true}\n')
    _touch(bundle / "submission_bundle_manifest.json", '{"bundle_ok": true}\n')

    out_json = step28 / "CAMERA_READY_CHECKLIST.json"
    out_md = step28 / "CAMERA_READY_CHECKLIST.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_camera_ready_checklist.py",
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
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("camera_ready_ok", False)) is True
