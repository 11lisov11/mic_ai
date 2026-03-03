from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_submission_handoff_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28" / "tagh"
    ieee = tmp_path / "ieee"
    bundle = ieee / "submission_bundle" / "tagh"

    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.json", '{"status":{"dossier_ok": true}}\n')
    _touch(
        bundle / "submission_bundle_manifest.json",
        '{"bundle_ok": true, "archives":{"zip":"bundle.zip","tar_gz":"bundle.tar.gz"}}\n',
    )
    _touch(bundle / "submission_bundle_manifest.md")

    out_json = step28 / "IEEE_SUBMISSION_HANDOFF.json"
    out_md = step28 / "IEEE_SUBMISSION_HANDOFF.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_submission_handoff.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee),
        "--tag",
        "tagh",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("handoff_ready", False)) is True

