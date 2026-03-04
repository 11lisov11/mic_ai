from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_integration_pipeline_dry_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "integration"
    cmd = [
        sys.executable,
        "tools/run_integration_pipeline.py",
        "--out-root",
        str(out_root),
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    reports = list(out_root.rglob("integration_pipeline_report.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert bool(payload.get("dry_run", False)) is True
    assert bool(payload.get("all_ok", False)) is True
    steps = list(payload.get("steps", []))
    assert len(steps) == 3
    assert any("tools/step27_pipeline.py" in " ".join(map(str, s.get("cmd", []))) for s in steps)
    assert any("tools/robust_motor_hardening.py" in " ".join(map(str, s.get("cmd", []))) for s in steps)
    assert any("tools/reproduce_ieee_step28.py" in " ".join(map(str, s.get("cmd", []))) for s in steps)

