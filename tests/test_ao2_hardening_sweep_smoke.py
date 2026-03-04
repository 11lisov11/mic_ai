from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_ao2_hardening_sweep_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "ao2_hardening"
    cmd = [
        sys.executable,
        "tools/ao2_hardening_sweep.py",
        "--out-dir",
        str(out_dir),
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--stage1-trials",
        "3",
        "--stage2-topk",
        "2",
        "--search-seed",
        "12345",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert (out_dir / "ao2_stage1_rank.csv").exists()
    assert (out_dir / "ao2_stage2_rank.csv").exists()
    assert (out_dir / "ao2_shortlist_top3.csv").exists()
    assert (out_dir / "ao2_selected_candidate_v2.json").exists()
    assert (out_dir / "ao2_hardening_summary_v2.json").exists()
    assert (out_dir / "ao2_hardening_summary_v2.md").exists()

    shortlist = pd.read_csv(out_dir / "ao2_shortlist_top3.csv")
    assert not shortlist.empty
    payload = json.loads((out_dir / "ao2_hardening_summary_v2.json").read_text(encoding="utf-8"))
    assert str(payload.get("motor", "")) == "ao2"
    assert int(payload.get("shortlist_count", 0)) >= 1

