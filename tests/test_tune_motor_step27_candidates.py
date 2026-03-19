from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tune_motor_step27 import _load_custom_candidates, _pass, _score


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
        "idle_blend": 1.0,
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
                    "idle_blend": "0.2",
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
    assert row["idle_blend"] == 0.2
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


def test_score_penalizes_envelope_fail_when_enabled() -> None:
    aggregate_green_but_envelope_red = {
        "avg_power_saving_pct": 1.0,
        "avg_eta_gain_pct": 1.0,
        "err_failures": 1.0,
        "start_stop_power_saving_pct": 1.0,
        "worst_current_peak_ratio": 1.0,
        "worst_current_mean_ratio": 1.0,
        "envelope_all_rows_pass": False,
        "envelope_fail_count": 3,
        "envelope_scenario_fail_count": 2,
        "envelope_gap_total": 1.5,
        "envelope_err_fail_count": 1,
    }
    weaker_but_envelope_green = {
        "avg_power_saving_pct": 0.1,
        "avg_eta_gain_pct": 0.1,
        "err_failures": 1.0,
        "start_stop_power_saving_pct": 0.1,
        "worst_current_peak_ratio": 1.0,
        "worst_current_mean_ratio": 1.0,
        "envelope_all_rows_pass": True,
        "envelope_fail_count": 0,
        "envelope_scenario_fail_count": 0,
        "envelope_gap_total": 0.0,
        "envelope_err_fail_count": 0,
    }

    score_red = _score(
        aggregate_green_but_envelope_red,
        min_power=0.0,
        min_eta=0.0,
        max_eta=25.0,
        max_err=2.0,
        min_start_stop=-0.5,
        max_start_stop=20.0,
        max_peak_ratio=1.3,
        max_mean_ratio=1.2,
        require_envelope_pass=True,
    )
    score_green = _score(
        weaker_but_envelope_green,
        min_power=0.0,
        min_eta=0.0,
        max_eta=25.0,
        max_err=2.0,
        min_start_stop=-0.5,
        max_start_stop=20.0,
        max_peak_ratio=1.3,
        max_mean_ratio=1.2,
        require_envelope_pass=True,
    )

    assert score_green < score_red
    assert score_red >= 100000.0


def test_pass_requires_envelope_green_when_enabled() -> None:
    metrics = {
        "avg_power_saving_pct": 1.0,
        "avg_eta_gain_pct": 1.0,
        "err_failures": 1.0,
        "start_stop_power_saving_pct": 1.0,
        "worst_current_peak_ratio": 1.0,
        "worst_current_mean_ratio": 1.0,
        "envelope_all_rows_pass": False,
    }

    assert (
        _pass(
            metrics,
            min_power=0.0,
            min_eta=0.0,
            max_eta=25.0,
            max_err=2.0,
            min_start_stop=-0.5,
            max_start_stop=20.0,
            max_peak_ratio=1.3,
            max_mean_ratio=1.2,
            require_envelope_pass=False,
        )
        is True
    )
    assert (
        _pass(
            metrics,
            min_power=0.0,
            min_eta=0.0,
            max_eta=25.0,
            max_err=2.0,
            min_start_stop=-0.5,
            max_start_stop=20.0,
            max_peak_ratio=1.3,
            max_mean_ratio=1.2,
            require_envelope_pass=True,
        )
        is False
    )
