from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_submission_candidate_note_smoke(tmp_path: Path) -> None:
    step28 = tmp_path / "step28_pkg"
    ieee_root = tmp_path / "ieee"
    step28.mkdir(parents=True, exist_ok=True)
    ieee_root.mkdir(parents=True, exist_ok=True)

    _touch(
        step28 / "step28_ieee_summary.csv",
        (
            "mode,motor,controller,avg_power_saving_pct_mean,avg_power_saving_pct_min,avg_eta_gain_pct_mean,avg_eta_gain_pct_min,err_failures_max\n"
            "mode1,air56,MIC,0.6,0.6,0.1,0.1,0\n"
            "mode1,air56,FOC,0.0,0.0,0.0,0.0,0\n"
        ),
    )
    _touch(
        step28 / "submission_candidate_lock.json",
        '{"lock_ok": true, "aggregate_sha256": "abc123", "hashed_files_count": 10, "required_files_missing": []}\n',
    )
    _touch(ieee_root / "FINAL_CHECKLIST_AUTO.md", "- [x] ready_for_submission: `True`\n")

    out_md = ieee_root / "SUBMISSION_CANDIDATE.md"
    out_json = ieee_root / "SUBMISSION_CANDIDATE.json"

    cmd = [
        sys.executable,
        "tools/build_submission_candidate_note.py",
        "--step28-dir",
        str(step28),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        "smoke_candidate",
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert out_md.exists()
    assert out_json.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "candidate_tag: `smoke_candidate`" in text
    assert "ready_for_submission: `True`" in text
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("ready_for_submission", False)) is True
    assert str(payload.get("candidate_tag", "")) == "smoke_candidate"
