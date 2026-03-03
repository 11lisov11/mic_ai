from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_frozen_ci_baseline_regression(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench_ci_regression"
    compare_report = out_dir / "compare_report.json"
    cmd = [
        sys.executable,
        "-m",
        "mic_ai.tools.run_benchmark",
        "--env-config",
        "config/env_demo_true_motor1_nominal.py",
        "--out-dir",
        str(out_dir),
        "--mic-id-ref-low",
        "1.0",
        "--mic-id-ref-high",
        "1.4",
        "--mic-id-ref-speed-tol-rel",
        "0.05",
        "--mic-id-ref-omega-min",
        "0.1",
        "--scenarios",
        "speed_step:0.2",
        "--t-end",
        "0.4",
        "--dt",
        "0.002",
        "--window-frac",
        "0.5",
        "--error-tol-rel",
        "0.05",
        "--error-tol-abs",
        "0.0",
        "--min-power-saving-pct",
        "-100",
        "--no-require-err-ok",
        "--baseline-summary",
        "benchmarks/baseline_summary_ci.json",
        "--compare-max-err-rel",
        "0.1",
        "--compare-max-power-rel",
        "0.1",
        "--compare-report",
        str(compare_report),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "summary.json").exists()
    assert compare_report.exists()
    payload = json.loads(compare_report.read_text(encoding="utf-8"))
    assert bool(payload.get("passed", False)) is True
    assert list(payload.get("failures", [])) == []
