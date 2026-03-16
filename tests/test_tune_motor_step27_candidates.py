from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tune_motor_step27 import _load_custom_candidates


def _base_candidate() -> dict[str, object]:
    return {
        "tag": "baseline",
        "source": "config",
        "objective": "specific_power",
        "speed_tol_rel": 0.1,
        "speed_tol_abs": 0.0,
        "omega_min_pu": 0.1,
        "update_steps": 12,
        "dither_amp": 0.01,
        "bias_step": 0.005,
        "bias_max": 0.12,
        "shaft_eps": 10.0,
        "reset_decay": 0.98,
        "objective_clip": 10.0,
        "idle_enable": False,
        "idle_omega_pu": 0.05,
        "idle_action": -0.8,
        "idle_exit_boost_steps": 0,
        "idle_exit_action": 0.95,
        "idle_bias_decay": 0.96,
        "id_ref_alpha": 0.2,
        "delta_id_max": 0.08,
        "id_ref_gate_speed_tol_rel": 0.1,
        "id_ref_gate_min_scale": 0.1,
        "id_ref_gate_exponent": 1.0,
    }


def test_load_custom_candidates_from_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [
                {
                    "tag": "c1",
                    "source": "",
                    "objective": "p_in",
                    "update_steps": "18",
                    "idle_enable": "true",
                    "objective_clip": "None",
                    "id_ref_alpha": "0.31",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = _load_custom_candidates(path, base=_base_candidate())
    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == "c1"
    assert row["source"] == "candidates.json"
    assert row["objective"] == "p_in"
    assert row["update_steps"] == 18
    assert row["idle_enable"] is True
    assert row["objective_clip"] is None
    assert row["id_ref_alpha"] == 0.31
    assert row["delta_id_max"] == 0.08


def test_load_custom_candidates_from_object_payload_assigns_default_tag(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps({"candidates": [{"tag": "", "id_ref_alpha": 0.27, "idle_enable": 0}]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = _load_custom_candidates(path, base=_base_candidate())
    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == "custom_001"
    assert row["source"] == "config"
    assert row["idle_enable"] is False
    assert row["id_ref_alpha"] == 0.27
