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


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    p2s = [0.05, 0.10, 0.15, 0.20, 0.23]
    m2s = [0.3, 0.6, 0.9, 1.2, 1.4]
    i1s = [0.22, 0.27, 0.33, 0.41, 0.48]
    n2s = [1384, 1383, 1382, 1381, 1380]
    etas = [62.0, 70.0, 75.0, 77.0, 76.5]
    coss = [0.32, 0.52, 0.68, 0.79, 0.80]
    for policy in ("FOC", "MIC"):
        for p2, m2, i1, n2, eta, cos in zip(p2s, m2s, i1s, n2s, etas, coss):
            rows.append(
                {
                    "policy": policy,
                    "p2_kw": p2,
                    "m2": m2,
                    "i_rms": i1,
                    "n2_rpm": n2,
                    "eta_pct": eta,
                    "cos_phi": cos,
                    "p_el_pos": p2 * 1000.0 / max(eta / 100.0, 1e-6),
                }
            )
    return rows


def test_build_theory_validation_reports_smoke(tmp_path: Path) -> None:
    tag = "tag_test"
    passport_root = tmp_path / "passport"
    out_root = tmp_path / "theory_validation"
    for motor in ("air56", "al31", "ao2"):
        _write_csv(passport_root / tag / "raw" / motor / "working_characteristics_filtered.csv", _rows())

    cmd = [
        sys.executable,
        "tools/build_theory_validation_reports.py",
        "--tag",
        tag,
        "--passport-root",
        str(passport_root),
        "--out-root",
        str(out_root),
    ]
    subprocess.run(cmd, check=True)

    summary_json = out_root / tag / "theory_validation_summary.json"
    summary_csv = out_root / tag / "theory_validation_summary.csv"
    summary_md = out_root / tag / "theory_validation_summary.md"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert summary_md.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["all_passed"] is True
    assert len(payload["rows"]) == 3

