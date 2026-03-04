from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_physical_config_policy_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "policy" / "physical_config_policy_3motors.json"
    out_md = tmp_path / "policy" / "physical_config_policy_3motors.md"
    cmd = [
        sys.executable,
        "tools/build_physical_config_policy.py",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True)
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    motors = dict(payload.get("motors", {}))
    assert {"air56", "al31", "ao2"}.issubset(set(motors.keys()))
    for key in ("air56", "al31", "ao2"):
        item = dict(motors[key])
        assert str(item.get("module", "")).startswith("config.env_research_")
        assert isinstance(item.get("loss_model", {}), dict)
        assert isinstance(item.get("sim_setup", {}), dict)

