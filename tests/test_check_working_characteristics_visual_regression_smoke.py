from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _csv_rows() -> str:
    return "\n".join(
        [
            "policy,P2_kW,M2_Nm,I1_A,n2_rpm,eta_pct,cosphi",
            "FOC,0.00,0.00,0.26,1380,0.0,0.17",
            "FOC,0.05,0.35,0.30,1379,58.0,0.52",
            "FOC,0.10,0.70,0.36,1377,72.0,0.72",
            "FOC,0.15,1.05,0.42,1374,78.0,0.80",
            "FOC,0.20,1.40,0.48,1370,76.0,0.79",
            "MIC_AI,0.00,0.00,0.25,1380,0.0,0.16",
            "MIC_AI,0.05,0.35,0.29,1379,59.0,0.53",
            "MIC_AI,0.10,0.70,0.35,1377,73.0,0.73",
            "MIC_AI,0.15,1.05,0.41,1374,79.0,0.81",
            "MIC_AI,0.20,1.40,0.47,1370,77.0,0.80",
        ]
    )


def test_check_working_characteristics_visual_regression_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "working_characteristics_air56_foc_mic_table.csv"
    csv_path.write_text(_csv_rows() + "\n", encoding="utf-8")
    out_json = tmp_path / "visual_regression.json"
    out_md = tmp_path / "visual_regression.md"
    cmd = [
        sys.executable,
        "tools/check_working_characteristics_visual_regression.py",
        "--csv",
        str(csv_path),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("passed", False)) is True
    assert bool(dict(payload.get("axis_consistency", {})).get("ok", False)) is True
    assert out_md.exists()

