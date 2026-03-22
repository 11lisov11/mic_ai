from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import tune_motor_step27 as tune_mod
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


def test_score_and_pass_can_require_min_seed_thresholds() -> None:
    weak_worst_seed = {
        "avg_power_saving_pct": 1.0,
        "avg_eta_gain_pct": 1.0,
        "err_failures": 0.0,
        "start_stop_power_saving_pct": 1.0,
        "worst_current_peak_ratio": 1.0,
        "worst_current_mean_ratio": 1.0,
        "avg_power_saving_pct_min_seed": -0.01,
        "avg_eta_gain_pct_min_seed": -0.02,
        "err_failures_max_seed": 1.0,
        "start_stop_power_saving_pct_min_seed": -0.25,
        "envelope_all_rows_pass": True,
    }
    strict_green = {
        **weak_worst_seed,
        "avg_power_saving_pct_min_seed": 0.10,
        "avg_eta_gain_pct_min_seed": 0.05,
        "err_failures_max_seed": 0.0,
        "start_stop_power_saving_pct_min_seed": 0.1,
    }

    score_red = _score(
        weak_worst_seed,
        min_power=0.0,
        min_eta=0.0,
        max_eta=25.0,
        max_err=2.0,
        min_start_stop=-0.5,
        max_start_stop=20.0,
        max_peak_ratio=1.3,
        max_mean_ratio=1.2,
        require_envelope_pass=False,
        min_power_min_seed=0.0,
        min_eta_min_seed=0.0,
        max_err_max_seed=0.5,
        min_start_stop_min_seed=-0.1,
    )
    score_green = _score(
        strict_green,
        min_power=0.0,
        min_eta=0.0,
        max_eta=25.0,
        max_err=2.0,
        min_start_stop=-0.5,
        max_start_stop=20.0,
        max_peak_ratio=1.3,
        max_mean_ratio=1.2,
        require_envelope_pass=False,
        min_power_min_seed=0.0,
        min_eta_min_seed=0.0,
        max_err_max_seed=0.5,
        min_start_stop_min_seed=-0.1,
    )

    assert score_green < score_red
    assert (
        _pass(
            weak_worst_seed,
            min_power=0.0,
            min_eta=0.0,
            max_eta=25.0,
            max_err=2.0,
            min_start_stop=-0.5,
            max_start_stop=20.0,
            max_peak_ratio=1.3,
            max_mean_ratio=1.2,
            require_envelope_pass=False,
            min_power_min_seed=0.0,
            min_eta_min_seed=0.0,
            max_err_max_seed=0.5,
            min_start_stop_min_seed=-0.1,
        )
        is False
    )
    assert (
        _pass(
            strict_green,
            min_power=0.0,
            min_eta=0.0,
            max_eta=25.0,
            max_err=2.0,
            min_start_stop=-0.5,
            max_start_stop=20.0,
            max_peak_ratio=1.3,
            max_mean_ratio=1.2,
            require_envelope_pass=False,
            min_power_min_seed=0.0,
            min_eta_min_seed=0.0,
            max_err_max_seed=0.5,
            min_start_stop_min_seed=-0.1,
        )
        is True
    )


def test_main_accepts_explicit_checkpoint_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep002.pth"
    ckpt.write_bytes(b"test")
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(json.dumps([{"tag": "only"}], indent=2), encoding="utf-8")
    out_dir = tmp_path / "out"

    calls: dict[str, object] = {"load_env_require_agent": None, "load_agent_path": None}

    def _fake_load_env_and_agent(config_path: str, *, foc_disable_lut: bool, require_agent: bool, motor_key: str | None = None, checkpoint_registry_path: str | None = None):
        calls["load_env_require_agent"] = require_agent
        env_cfg = object()
        return env_cfg, None, None

    def _fake_load_agent(path: Path):
        calls["load_agent_path"] = Path(path)
        return object()

    def _fake_supervisor_from_env(env_cfg: object):
        return None

    def _fake_id_ref_eval_params(env_cfg: object):
        return {
            "id_ref_alpha": 0.2,
            "delta_id_max": 0.08,
            "id_ref_gate_speed_tol_rel": 0.1,
            "id_ref_gate_min_scale": 0.1,
            "id_ref_gate_exponent": 1.0,
        }

    def _fake_eval_candidate(**kwargs):
        return {
            "avg_power_saving_pct": 1.0,
            "avg_eta_gain_pct": 0.5,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 1.0,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.5,
            "avg_eta_gain_pct_min_seed": 0.1,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 1.0,
            "envelope_all_rows_pass": True,
            "envelope_fail_count": 0,
            "envelope_scenario_fail_count": 0,
            "envelope_gap_total": 0.0,
            "envelope_power_gap": 0.0,
            "envelope_eta_gap": 0.0,
            "envelope_peak_gap": 0.0,
            "envelope_mean_gap": 0.0,
            "envelope_err_fail_count": 0,
            "envelope_summary_rows": [],
        }

    monkeypatch.setattr(tune_mod, "_load_env_and_agent", _fake_load_env_and_agent)
    monkeypatch.setattr(tune_mod, "_load_agent", _fake_load_agent)
    monkeypatch.setattr(tune_mod, "_supervisor_from_env", _fake_supervisor_from_env)
    monkeypatch.setattr(tune_mod, "_id_ref_eval_params", _fake_id_ref_eval_params)
    monkeypatch.setattr(tune_mod, "_eval_candidate", _fake_eval_candidate)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_motor_step27.py",
            "--motor",
            "air56",
            "--checkpoint-path",
            str(ckpt),
            "--candidate-json",
            str(candidate_json),
            "--candidate-json-mode",
            "replace",
            "--out-dir",
            str(out_dir),
        ],
    )

    tune_mod.main()

    assert calls["load_env_require_agent"] is False
    assert calls["load_agent_path"] == ckpt.resolve()
    summary = json.loads((out_dir / "air56_tuning_summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"] == str(ckpt.resolve())
    assert summary["checkpoint_source"] == "explicit"
