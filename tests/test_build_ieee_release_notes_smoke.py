from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_ieee_release_notes_smoke(tmp_path: Path) -> None:
    tag = "tagx"
    step28 = tmp_path / "step28" / tag
    ieee_root = tmp_path / "ieee"
    bundle = ieee_root / "submission_bundle" / tag

    _write_csv(
        step28 / "step28_ieee_summary.csv",
        [
            {
                "mode": "mode1",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 0.9,
                "avg_power_saving_pct_min": 0.1,
                "avg_eta_gain_pct_mean": 0.07,
                "avg_eta_gain_pct_min": 0.01,
                "err_failures_max": 2.0,
                "start_stop_power_saving_pct_mean": 2.1,
                "start_stop_power_saving_pct_min": -0.2,
            },
            {
                "mode": "mode2",
                "controller": "MIC",
                "avg_power_saving_pct_mean": 0.8,
                "avg_power_saving_pct_min": 0.05,
                "avg_eta_gain_pct_mean": 0.06,
                "avg_eta_gain_pct_min": 0.01,
                "err_failures_max": 2.0,
                "start_stop_power_saving_pct_mean": 1.8,
                "start_stop_power_saving_pct_min": -0.3,
            },
        ],
    )
    _touch(step28 / "VERIFY_SUBMISSION_CANDIDATE.json", '{"verification_ok": true}\n')
    _touch(step28 / "IEEE_SUBMISSION_DOSSIER.json", '{"status":{"dossier_ok": true}}\n')
    _touch(step28 / "RELEASE_COMMIT_MANIFEST.json", '{"manifest_ok": true}\n')
    _touch(step28 / "STEP28_REGRESSION_GUARD.json", '{"ok": true}\n')
    _touch(bundle / "submission_bundle_manifest.json", '{"bundle_ok": true, "archives":{"zip":"a.zip","tar_gz":"a.tgz"}}\n')

    out_json = step28 / "IEEE_RELEASE_NOTES.json"
    out_md = step28 / "IEEE_RELEASE_NOTES.md"
    cmd = [
        sys.executable,
        "tools/build_ieee_release_notes.py",
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
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(payload.get("mic_rows_count", 0)) == 2
    assert bool(payload.get("release_note_ready", False)) is True
    assert bool(payload.get("strict_ready", False)) is True
