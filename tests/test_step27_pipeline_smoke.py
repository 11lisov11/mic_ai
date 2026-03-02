from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_step27_pipeline_smoke_rule_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "step27_smoke"
    cmd = [
        sys.executable,
        "tools/step27_pipeline.py",
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--skip-air56-tune",
        "--mic-mode",
        "rule",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    expected = [
        "step27_per_seed_metrics.csv",
        "step27_stats_motor_controller.csv",
        "step27_final_pi_vs_foc_vs_mic.csv",
        "step27_air56_acceptance.json",
        "step27_reproducibility.json",
        "step27_report.md",
    ]
    for name in expected:
        assert (out_dir / name).exists(), f"missing artifact: {name}"

