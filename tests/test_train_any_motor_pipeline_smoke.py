from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_any_motor_pipeline_dry_smoke(tmp_path: Path) -> None:
    passport = tmp_path / "passport_motor_x.json"
    passport.write_text(
        json.dumps(
            {
                "motor_key": "motor_x",
                "P_kW": 0.55,
                "U_ll": 380.0,
                "I_n": 1.7,
                "cos_phi_n": 0.74,
                "eta_n": 0.78,
                "f_n": 50.0,
                "p": 2,
                "n_rated": 1390.0,
                "connection": "Y",
                "J": 0.02,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "onboard_out"
    cfg_root = tmp_path / "generated_cfg"
    cmd = [
        sys.executable,
        "tools/train_any_motor_pipeline.py",
        "--passport-json",
        str(passport),
        "--motor-key",
        "motor_x",
        "--out-dir",
        str(out_root),
        "--generated-config-dir",
        str(cfg_root),
        "--run-tag",
        "smoke",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    report = out_root / "smoke" / "any_motor_onboarding_report.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert bool(payload.get("dry_run", False)) is True
    assert bool(payload.get("all_ok", False)) is True

    config_path = Path(str(payload.get("generated_config_path", "")))
    assert config_path.exists()

    steps = list(payload.get("steps", []))
    names = [str(s.get("name", "")) for s in steps if isinstance(s, dict)]
    assert "generate_config" in names
    assert "train_policy" in names
    assert "validate_benchmarks" in names

    benchmark_plan = out_root / "smoke" / "benchmark_validation_plan.json"
    assert benchmark_plan.exists()
    plan_rows = json.loads(benchmark_plan.read_text(encoding="utf-8"))
    motors = sorted({str(row.get("motor", "")) for row in plan_rows if isinstance(row, dict)})
    assert motors == ["air56", "al31"]
