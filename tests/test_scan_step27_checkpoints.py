from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scan_step27_checkpoints import _select_candidate


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


def test_select_candidate_by_index(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [
                {"tag": "c0", "id_ref_alpha": 0.21},
                {"tag": "c1", "id_ref_alpha": 0.31, "objective": "p_in"},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row, total = _select_candidate(path, base=_base_candidate(), candidate_index=1, candidate_tag="")
    assert total == 2
    assert row["tag"] == "c1"
    assert row["objective"] == "p_in"
    assert row["id_ref_alpha"] == 0.31
    assert row["delta_id_max"] == 0.08


def test_select_candidate_by_tag(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {"candidates": [{"tag": "low"}, {"tag": "keep_me", "bias_max": 0.22}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row, total = _select_candidate(path, base=_base_candidate(), candidate_index=0, candidate_tag="keep_me")
    assert total == 2
    assert row["tag"] == "keep_me"
    assert row["bias_max"] == 0.22
    assert row["objective"] == "specific_power"


def test_select_candidate_rejects_missing_tag(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"tag": "c0"}], ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="was not found"):
        _select_candidate(path, base=_base_candidate(), candidate_index=0, candidate_tag="missing")
