from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_step27_extended_repro_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "extended_repro"
    cmd = [
        sys.executable,
        "tools/run_step27_extended_repro.py",
        "--out-dir",
        str(out_dir),
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--perturb-levels",
        "0.2",
        "--mic-mode",
        "rule",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    per_seed = out_dir / "step27_extended_per_seed_metrics.csv"
    stats = out_dir / "step27_extended_stats.csv"
    stress = out_dir / "step27_extended_stress_sweep.csv"
    manifest = out_dir / "step27_extended_manifest.json"
    report = out_dir / "step27_extended_report.md"

    assert per_seed.exists()
    assert stats.exists()
    assert stress.exists()
    assert manifest.exists()
    assert report.exists()

    stats_df = pd.read_csv(stats)
    assert not stats_df.empty
    run_tags = set(stats_df["run_tag"].astype(str).tolist())
    assert "baseline" in run_tags
    assert "perturb_0p2" in run_tags
