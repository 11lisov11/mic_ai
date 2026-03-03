from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_check_ieee_manuscript_consistency_smoke(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manuscript = root / "paper" / "ieee_2026" / "manuscript.md"
    ref1 = root / "paper" / "ieee_2026" / "fig" / "fig2_pi_foc_mic_power.pdf"
    ref2 = root / "paper" / "ieee_2026" / "data" / "release" / "x" / "tables" / "step28_ieee_summary.csv"
    _touch(ref1)
    _touch(ref2)
    _touch(
        manuscript,
        (
            "Figure ref: `paper/ieee_2026/fig/fig2_pi_foc_mic_power.pdf`\n"
            "Table ref: `paper/ieee_2026/data/release/x/tables/step28_ieee_summary.csv`\n"
            "See Fig. 2 and Table 1.\n"
        ),
    )

    out_json = manuscript.parent / "MANUSCRIPT_CONSISTENCY_REPORT.json"
    out_md = manuscript.parent / "MANUSCRIPT_CONSISTENCY_REPORT.md"
    cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_consistency.py",
        "--manuscript",
        str(manuscript),
        "--repo-root",
        str(root),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("ok", False)) is True
    assert int(payload.get("paths_missing", 0)) == 0
