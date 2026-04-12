from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import diagnose_motor_nominal_consistency as diag


def test_build_diagnosis_payload_accepts_calibrated_ao2_live_config() -> None:
    module = diag._resolve_config_module("env_research_ao2_32_4_3kw")
    payload = diag.build_diagnosis_payload(
        module,
        run_probes=False,
        t_end=2.0,
        dt=1e-3,
        steady_window_frac=0.25,
        probe_load_factors=(0.25, 1.0),
    )

    cons = payload["consistency"]
    assert cons["config_load_to_nominal_torque_ratio"] == 0.25
    assert cons["rough_foc_torque_capacity_to_nominal_ratio"] > 1.5
    assert list(payload["warnings"]) == []


def test_main_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "diag"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_motor_nominal_consistency.py",
            "--config",
            "env_research_ao2_32_4_3kw",
            "--out-dir",
            str(out_dir),
            "--skip-probes",
        ],
    )
    diag.main()

    json_path = out_dir / "motor_nominal_consistency.json"
    md_path = out_dir / "motor_nominal_consistency.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config_module"].endswith("env_research_ao2_32_4_3kw")
    assert "rough_foc_torque_capacity_to_nominal_ratio" in payload["consistency"]
