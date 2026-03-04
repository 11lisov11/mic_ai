from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eval_cross_motor_generalization_dry_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "cross_gen"
    cmd = [
        sys.executable,
        "tools/eval_cross_motor_generalization.py",
        "--mode",
        "heldout",
        "--source-motors",
        "air56,al31",
        "--target-motors",
        "ao2",
        "--seeds",
        "101,202",
        "--scenarios",
        "speed_step",
        "--dry-run",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    manifest = json.loads((out_dir / "cross_motor_generalization_manifest.json").read_text(encoding="utf-8"))
    assert bool(manifest.get("dry_run", False)) is True
    pair_plan = json.loads((out_dir / "cross_motor_generalization_pair_plan.json").read_text(encoding="utf-8"))
    # heldout: sources air56/al31, target ao2, seeds 101/202 => 4 + native ao2->ao2 (2) = 6
    assert len(pair_plan) == 6
    summary = json.loads((out_dir / "cross_motor_generalization_summary.json").read_text(encoding="utf-8"))
    assert summary == []
    gap = json.loads((out_dir / "cross_motor_generalization_gap_vs_native.json").read_text(encoding="utf-8"))
    assert gap == []

