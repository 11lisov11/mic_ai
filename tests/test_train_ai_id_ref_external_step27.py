from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mic_ai.ai.train_ai_id_ref import (
    _adapt_checkpoint_state_dict_for_model,
    _promote_external_step27_checkpoint,
    _run_external_step27_selection,
    build_env,
    build_feature_keys,
)


def test_run_external_step27_selection_promotes_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    selected_ckpt = eval_dir / "actor_ep003.pth"
    selected_ckpt.write_bytes(b"selected")
    init_ckpt = tmp_path / "init_actor.pth"
    init_ckpt.write_bytes(b"init")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(json.dumps([{"tag": "base_current"}], indent=2), encoding="utf-8")

    def _fake_scan_checkpoints(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        assert (eval_dir / "actor_ep_init.pth").exists()
        assert (eval_dir / "actor_ep_init.pth").read_bytes() == b"init"
        assert kwargs["feature_keys"] == ["omega_norm", "eta_episode_norm"]
        assert kwargs["min_avg_power_saving_pct"] == pytest.approx(0.5)
        assert kwargs["min_avg_eta_gain_pct"] == pytest.approx(0.0)
        assert kwargs["max_avg_eta_gain_pct"] == pytest.approx(25.0)
        assert kwargs["max_err_failures"] == pytest.approx(2.0)
        assert kwargs["min_start_stop_saving_pct"] == pytest.approx(-0.5)
        assert kwargs["max_start_stop_saving_pct"] == pytest.approx(20.0)
        assert kwargs["max_worst_current_peak_ratio"] == pytest.approx(1.3)
        assert kwargs["max_worst_current_mean_ratio"] == pytest.approx(1.2)
        assert kwargs["use_envelope_acceptance"] is True
        assert str(kwargs["acceptance_envelopes"]).endswith("acceptance_envelopes_3motors.json")
        assert kwargs["min_avg_power_saving_pct_min_seed"] == pytest.approx(0.5)
        assert kwargs["min_avg_eta_gain_pct_min_seed"] == pytest.approx(0.0)
        assert kwargs["max_err_failures_max_seed"] == pytest.approx(2.0)
        assert kwargs["min_start_stop_saving_pct_min_seed"] == pytest.approx(-0.5)
        summary = {
            "best": {
                "checkpoint": str(selected_ckpt.resolve()),
                "checkpoint_name": selected_ckpt.name,
                "rank": 1,
                "score": 1.25,
                "acceptance_pass": False,
            }
        }
        (out_dir / "ao2_checkpoint_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "ao2_checkpoint_scan.json").write_text(json.dumps([summary["best"]], indent=2), encoding="utf-8")
        return summary

    monkeypatch.setattr("tools.scan_step27_checkpoints.scan_checkpoints", _fake_scan_checkpoints)

    payload = _run_external_step27_selection(
        run_dir=run_dir,
        motor="ao2",
        candidate_json=str(candidate_json),
        candidate_index=0,
        candidate_tag="",
        seeds="101,202,303",
        scenarios="speed_step,ramp,load_step,start_stop",
        seed_perturbation=True,
        seed_perturb_level=0.2,
        min_avg_power_saving_pct=0.5,
        min_avg_eta_gain_pct=0.0,
        max_avg_eta_gain_pct=25.0,
        max_err_failures=2.0,
        min_start_stop_saving_pct=-0.5,
        max_start_stop_saving_pct=20.0,
        max_worst_current_peak_ratio=1.3,
        max_worst_current_mean_ratio=1.2,
        use_envelope_acceptance=True,
        acceptance_envelopes="config/acceptance_envelopes_3motors.json",
        min_avg_power_saving_pct_min_seed=0.5,
        min_avg_eta_gain_pct_min_seed=0.0,
        max_err_failures_max_seed=2.0,
        min_start_stop_saving_pct_min_seed=-0.5,
        top_k=5,
        feature_keys=["omega_norm", "eta_episode_norm"],
        init_checkpoint=str(init_ckpt),
        include_init_checkpoint=True,
    )

    promoted = run_dir / "best_actor_step27.pth"
    assert promoted.exists()
    assert promoted.read_bytes() == b"selected"
    assert payload["selected_checkpoint"] == str(selected_ckpt.resolve())
    assert payload["promoted_checkpoint"] == str(promoted.resolve())
    assert payload["seeds"] == [101, 202, 303]
    assert payload["scenarios"] == ["speed_step", "ramp", "load_step", "start_stop"]
    assert payload["min_avg_power_saving_pct"] == pytest.approx(0.5)
    assert payload["min_avg_eta_gain_pct"] == pytest.approx(0.0)
    assert payload["max_avg_eta_gain_pct"] == pytest.approx(25.0)
    assert payload["max_err_failures"] == pytest.approx(2.0)
    assert payload["min_start_stop_saving_pct"] == pytest.approx(-0.5)
    assert payload["max_start_stop_saving_pct"] == pytest.approx(20.0)
    assert payload["max_worst_current_peak_ratio"] == pytest.approx(1.3)
    assert payload["max_worst_current_mean_ratio"] == pytest.approx(1.2)
    assert payload["use_envelope_acceptance"] is True
    assert payload["acceptance_envelopes"].endswith("acceptance_envelopes_3motors.json")
    assert payload["min_avg_power_saving_pct_min_seed"] == pytest.approx(0.5)
    assert payload["min_avg_eta_gain_pct_min_seed"] == pytest.approx(0.0)
    assert payload["max_err_failures_max_seed"] == pytest.approx(2.0)
    assert payload["min_start_stop_saving_pct_min_seed"] == pytest.approx(-0.5)
    assert payload["include_init_checkpoint"] is True
    assert payload["included_init_checkpoint"] == str((eval_dir / "actor_ep_init.pth").resolve())


def test_promote_external_step27_checkpoint_updates_registry_best(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    registry_best = ckpt_dir / "best_actor.pth"
    registry_best.write_bytes(b"train-best")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = run_dir / "best_actor_step27.pth"
    selected.write_bytes(b"step27-best")

    payload = _promote_external_step27_checkpoint(
        ckpt_dir=ckpt_dir,
        external_step27_selection={"promoted_checkpoint": str(selected.resolve())},
    )

    assert payload is not None
    assert registry_best.read_bytes() == b"step27-best"
    assert (ckpt_dir / "best_actor_train_internal.pth").read_bytes() == b"train-best"
    assert payload["registry_best_checkpoint"] == str(registry_best.resolve())


def test_build_env_propagates_reward_gate_and_terminal_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text(
        "\n".join(
            [
                "from config.env_demo_true_motor1 import *  # noqa: F401,F403",
                "ai_id_energy_gate_mode = 'soft'",
                "ai_id_energy_gate_min_scale = 0.15",
                "ai_id_energy_gate_exponent = 1.7",
                "ai_id_terminal_energy_bonus = 0.6",
                "ai_id_terminal_eta_target = 0.4",
                "ai_id_terminal_shaft_ratio_min = 0.8",
                "i_soft_limit = 0.9",
                "i_soft_penalty = 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode="ai_id_ref",
        w_speed=1.0,
        w_power=2.0,
        w_current=None,
        w_smooth=0.05,
        w_mag=0.0,
        w_shaft=0.0,
        w_eta=0.0,
        w_eta_episode=0.0,
        eta_clip=1.2,
        override_load_torque=True,
        override_omega_ref=True,
        ai_id_ref_relative=True,
        delta_id_max=0.1,
        id_ref_alpha=1.0,
        id_ref_rate_limit=None,
        ai_id_speed_tol=0.5,
        ai_id_speed_tol_rel=None,
        id_ref_gate_speed_tol=None,
        id_ref_gate_speed_tol_rel=None,
        id_ref_gate_min_scale=0.0,
        id_ref_gate_exponent=1.0,
        load_torque=None,
        omega_ref_override=None,
        feature_keys=["omega_norm"],
    )

    assert env.cfg.ai_id_energy_gate_mode == "soft"
    assert env.cfg.ai_id_energy_gate_min_scale == pytest.approx(0.15)
    assert env.cfg.ai_id_energy_gate_exponent == pytest.approx(1.7)
    assert env.cfg.ai_id_terminal_energy_bonus == pytest.approx(0.6)
    assert env.cfg.ai_id_terminal_eta_target == pytest.approx(0.4)
    assert env.cfg.ai_id_terminal_shaft_ratio_min == pytest.approx(0.8)
    assert env.cfg.i_soft_limit == pytest.approx(0.9)
    assert env.cfg.i_soft_penalty == pytest.approx(0.7)


def test_build_feature_keys_includes_episode_eta_only_when_requested() -> None:
    keys_default = build_feature_keys(include_energy_obs=True, include_episode_eta_obs=False)
    keys_episode = build_feature_keys(include_energy_obs=True, include_episode_eta_obs=True)
    assert "eta_episode_norm" not in keys_default
    assert "eta_episode_norm" in keys_episode


def test_adapt_checkpoint_state_dict_for_model_zero_pads_new_input_columns() -> None:
    state = {
        "actor_body.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "critic_body.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    model_state = {
        "actor_body.0.weight": torch.zeros(2, 5, dtype=torch.float32),
        "critic_body.0.weight": torch.zeros(2, 5, dtype=torch.float32),
    }

    adapted, adjusted = _adapt_checkpoint_state_dict_for_model(state, model_state)

    assert set(adjusted) == {"actor_body.0.weight", "critic_body.0.weight"}
    assert adapted["actor_body.0.weight"].shape == (2, 5)
    assert torch.equal(adapted["actor_body.0.weight"][:, :3], state["actor_body.0.weight"])
    assert torch.equal(adapted["actor_body.0.weight"][:, 3:], torch.zeros(2, 2))
