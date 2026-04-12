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
    _apply_env_reward_overrides,
    _apply_scenario_reward_overrides,
    _adapt_checkpoint_state_dict_for_model,
    _collect_underhorizon_scenarios,
    _curriculum_scale,
    _estimate_scenario_activation_steps,
    _infer_hidden_sizes_from_state_dict,
    _parse_hidden_sizes,
    _parse_scenario_reward_overrides,
    _parse_seed_scenario_reward_overrides,
    _select_episode_seed,
    _promote_external_step27_checkpoint,
    _run_external_step27_selection,
    build_env,
    build_feature_keys,
    train,
)
from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent


def test_estimate_scenario_activation_steps_matches_scenario_timings() -> None:
    assert _estimate_scenario_activation_steps("speed_step", t_end=2.0, dt=5e-4) == 400
    assert _estimate_scenario_activation_steps("load_step", t_end=2.0, dt=5e-4) == 1200
    assert _estimate_scenario_activation_steps("start_stop", t_end=2.0, dt=5e-4) == 800
    assert _estimate_scenario_activation_steps("ramp", t_end=2.0, dt=5e-4) == 2400
    assert _estimate_scenario_activation_steps("hold:0.8", t_end=2.0, dt=5e-4) == 0


def test_collect_underhorizon_scenarios_flags_short_episode_horizon() -> None:
    rows = _collect_underhorizon_scenarios(
        ["load_step", "speed_step", "load_step", "hold:0.8"],
        episode_steps=150,
        t_end=2.0,
        dt=5e-4,
    )
    assert rows == [
        {
            "scenario": "load_step",
            "episode_steps": 150,
            "required_steps": 1200,
            "episode_horizon_s": pytest.approx(0.075),
            "required_horizon_s": pytest.approx(0.6),
        },
        {
            "scenario": "speed_step",
            "episode_steps": 150,
            "required_steps": 400,
            "episode_horizon_s": pytest.approx(0.075),
            "required_horizon_s": pytest.approx(0.2),
        },
    ]


def test_select_episode_seed_cycles_over_failing_seed_list() -> None:
    assert _select_episode_seed(0, [202, 505]) == 202
    assert _select_episode_seed(1, [202, 505]) == 505
    assert _select_episode_seed(2, [202, 505]) == 202
    assert _select_episode_seed(3, [202, 505]) == 505
    assert _select_episode_seed(0, None) is None


def test_build_env_respects_scenario_functions_when_overrides_disabled() -> None:
    env = build_env(
        env_config_path="config/env_research_air56_025kw.py",
        episode_steps=200,
        control_mode="ai_id_ref",
        w_speed=1.0,
        w_power=1.0,
        w_current=None,
        w_smooth=0.05,
        w_mag=0.0,
        w_shaft=0.0,
        w_eta=0.0,
        w_eta_episode=0.0,
        eta_clip=1.2,
        override_load_torque=False,
        override_omega_ref=False,
        ai_id_ref_relative=False,
        delta_id_max=0.3,
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
    env.set_scenario("load_step")
    env.reset()
    t_end = float(getattr(env.base_env.env.sim, "t_end", 0.0))
    load_before = float(env.base_env.load_torque_func(0.0))
    load_after = float(env.base_env.load_torque_func(0.31 * t_end))
    omega_before = float(env.base_env.omega_ref_func(0.0))
    omega_after = float(env.base_env.omega_ref_func(0.31 * t_end))
    assert load_before == pytest.approx(0.0)
    assert load_after > 0.0
    assert omega_before > 0.0
    assert omega_after == pytest.approx(omega_before)


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
        assert kwargs["config_path"].endswith("env_backlog_ao2_nameplate_foc_tuned.py")
        assert kwargs["ai_control_mode"] == "ai_current"
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
        assert kwargs["resume"] is True
        assert kwargs["candidate_tags"] == ["base_current", "base_alt"]
        summary = {
            "best": {
                "checkpoint": str(selected_ckpt.resolve()),
                "checkpoint_name": selected_ckpt.name,
                "rank": 1,
                "score": 1.25,
                "acceptance_pass": False,
                "candidate_tag": "base_alt",
            }
        }
        (out_dir / "ao2_checkpoint_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "ao2_checkpoint_scan.json").write_text(json.dumps([summary["best"]], indent=2), encoding="utf-8")
        return summary

    monkeypatch.setattr("tools.scan_step27_checkpoints.scan_checkpoints", _fake_scan_checkpoints)

    payload = _run_external_step27_selection(
        run_dir=run_dir,
        motor="ao2",
        config_path="config/env_backlog_ao2_nameplate_foc_tuned.py",
        ai_control_mode="ai_current",
        candidate_json=str(candidate_json),
        candidate_index=0,
        candidate_tag="",
        candidate_tags="base_current,base_alt",
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
        resume=True,
    )

    promoted = run_dir / "best_actor_step27.pth"
    assert promoted.exists()
    assert promoted.read_bytes() == b"selected"
    assert payload["selected_checkpoint"] == str(selected_ckpt.resolve())
    assert payload["promoted_checkpoint"] == str(promoted.resolve())
    assert payload["ai_control_mode"] == "ai_current"
    assert payload["config_path"].endswith("env_backlog_ao2_nameplate_foc_tuned.py")
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
    assert payload["resume"] is True
    assert payload["include_init_checkpoint"] is True
    assert payload["included_init_checkpoint"] == str((eval_dir / "actor_ep_init.pth").resolve())


def test_run_external_step27_selection_allows_missing_candidate_json_for_non_id_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    selected_ckpt = eval_dir / "actor_ep000.pth"
    selected_ckpt.write_bytes(b"selected")

    def _fake_scan_checkpoints(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        assert kwargs["config_path"].endswith("env_demo_true_motor1.py")
        assert kwargs["ai_control_mode"] == "foc_assist"
        assert kwargs["candidate_json"] == ""
        summary = {
            "best": {
                "checkpoint": str(selected_ckpt.resolve()),
                "checkpoint_name": selected_ckpt.name,
                "rank": 1,
                "score": 0.5,
                "acceptance_pass": False,
            }
        }
        (out_dir / "air56_checkpoint_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "air56_checkpoint_scan.json").write_text(json.dumps([summary["best"]], indent=2), encoding="utf-8")
        return summary

    monkeypatch.setattr("tools.scan_step27_checkpoints.scan_checkpoints", _fake_scan_checkpoints)

    payload = _run_external_step27_selection(
        run_dir=run_dir,
        motor="air56",
        config_path="config/env_demo_true_motor1.py",
        ai_control_mode="foc_assist",
        candidate_json=None,
        candidate_index=0,
        candidate_tag="",
        candidate_tags="",
        seeds="101",
        scenarios="speed_step",
        seed_perturbation=True,
        seed_perturb_level=0.2,
        min_avg_power_saving_pct=0.0,
        min_avg_eta_gain_pct=0.0,
        max_avg_eta_gain_pct=25.0,
        max_err_failures=2.0,
        min_start_stop_saving_pct=-0.5,
        max_start_stop_saving_pct=20.0,
        max_worst_current_peak_ratio=1.3,
        max_worst_current_mean_ratio=1.2,
        use_envelope_acceptance=True,
        acceptance_envelopes="config/acceptance_envelopes_3motors.json",
        min_avg_power_saving_pct_min_seed=0.0,
        min_avg_eta_gain_pct_min_seed=0.0,
        max_err_failures_max_seed=2.0,
        min_start_stop_saving_pct_min_seed=-0.5,
        top_k=5,
        feature_keys=["omega_norm"],
        init_checkpoint=None,
        include_init_checkpoint=False,
        resume=False,
    )

    assert payload["candidate_json"] is None
    assert payload["config_path"].endswith("env_demo_true_motor1.py")
    assert payload["ai_control_mode"] == "foc_assist"
    assert Path(str(payload["promoted_checkpoint"])).exists()
    assert payload["candidate_tags"] == []


def test_train_normalizes_none_candidate_json_for_non_id_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "init_actor.pth"
    agent = PPOVoltageAgent(feature_keys=["omega_norm"], action_dim=1, device="cpu", hidden_sizes=(64, 64))
    torch.save(agent.net.state_dict(), ckpt)

    seen: dict[str, object] = {}

    def _fake_build_env(*args, **kwargs):
        class _DummyEnv:
            def __init__(self) -> None:
                self.cfg = type("Cfg", (), {"ai_id_terminal_energy_bonus": 0.0, "omega_ref_max": 1.0, "omega_ref": 1.0})()
                self.base_env = type("BaseEnv", (), {"load_torque_func": staticmethod(lambda _t: 0.0)})()
                self._omega_nominal = 1.0
                self._train_env_cfg = None

            def reset(self):
                return {"omega": 0.0, "omega_ref": 0.0}

            def set_scenario(self, _name: str) -> None:
                return None

            def step(self, _action):
                return {"omega": 0.0, "omega_ref": 0.0}, 0.0, True, {}

            def episode_metrics(self):
                return {"steps": 1, "mean_speed_error": 0.0, "mean_p_in_pos": 0.0, "mean_p_shaft_pos": 0.0, "mean_p_shaft_target_pos": 0.0, "mean_eta_inst": 0.0, "eta_energy": 0.0, "mean_current_rms": 0.0, "action_norm": 0.0}

        return _DummyEnv()

    def _fake_selection(**kwargs):
        seen["candidate_json"] = kwargs["candidate_json"]
        seen["candidate_tags"] = kwargs["candidate_tags"]
        return {
            "promoted_checkpoint": str(tmp_path / "best_actor_step27.pth"),
            "selected_checkpoint": str(tmp_path / "best_actor_step27.pth"),
        }

    monkeypatch.setattr("mic_ai.ai.train_ai_id_ref.build_env", _fake_build_env)
    monkeypatch.setattr("mic_ai.ai.train_ai_id_ref._run_external_step27_selection", _fake_selection)
    monkeypatch.setattr("mic_ai.ai.train_ai_id_ref._promote_external_step27_checkpoint", lambda **kwargs: kwargs.get("external_step27_selection"))

    train(
        env_config="config/env_demo_true_motor1.py",
        episodes=1,
        episode_steps=1,
        control_mode="foc_assist",
        w_speed=1.0,
        w_power=1.0,
        w_current=None,
        w_smooth=0.0,
        w_mag=0.0,
        w_shaft=0.0,
        w_eta=0.0,
        w_eta_episode=0.0,
        eta_clip=1.2,
        id_ref_alpha=1.0,
        id_ref_rate_limit=None,
        ai_id_speed_tol=0.5,
        ai_id_speed_tol_rel=None,
        id_ref_gate_speed_tol=None,
        id_ref_gate_speed_tol_rel=None,
        id_ref_gate_min_scale=0.0,
        id_ref_gate_exponent=1.0,
        fast=True,
        time_budget_min=None,
        override_load_torque=True,
        override_omega_ref=True,
        ai_id_ref_relative=True,
        delta_id_max=0.1,
        load_torque=None,
        omega_ref_override=None,
        scenarios=[],
        scenario_sample="random",
        omega_ref_range=None,
        load_torque_range=None,
        seed=123,
        sigma_start=0.1,
        sigma_end=0.1,
        sigma_decay_episodes=1,
        power_warmup_episodes=0,
        power_ramp_episodes=0,
        energy_warmup_episodes=0,
        energy_ramp_episodes=0,
        eval_interval=0,
        eval_scenarios="speed_step",
        eval_dt=None,
        eval_t_end=None,
        eval_window_frac=0.25,
        eval_error_tol_rel=0.05,
        eval_error_tol_abs=0.0,
        eval_use_total_power=False,
        include_energy_obs=False,
        include_episode_eta_obs=False,
        update_every_episodes=1,
        lr=1e-4,
        entropy_coef=0.0,
        actor_anchor_coef=0.0,
        external_step27_select=True,
        external_step27_motor="air56",
        external_step27_candidate_json="None",
        external_step27_candidate_index=0,
        external_step27_candidate_tag="",
        external_step27_candidate_tags="",
        external_step27_seeds="101",
        external_step27_scenarios="speed_step",
        external_step27_seed_perturbation=True,
        external_step27_seed_perturb_level=0.2,
        external_step27_min_avg_power_saving_pct=0.0,
        external_step27_min_avg_eta_gain_pct=0.0,
        external_step27_max_avg_eta_gain_pct=25.0,
        external_step27_max_err_failures=2.0,
        external_step27_min_start_stop_saving_pct=-0.5,
        external_step27_max_start_stop_saving_pct=20.0,
        external_step27_max_worst_current_peak_ratio=1.3,
        external_step27_max_worst_current_mean_ratio=1.2,
        external_step27_use_envelope_acceptance=True,
        external_step27_acceptance_envelopes="config/acceptance_envelopes_3motors.json",
        external_step27_min_avg_power_saving_pct_min_seed=0.0,
        external_step27_min_avg_eta_gain_pct_min_seed=0.0,
        external_step27_max_err_failures_max_seed=2.0,
        external_step27_min_start_stop_saving_pct_min_seed=-0.5,
        external_step27_top_k=3,
        external_step27_include_init_checkpoint=False,
        external_step27_resume=False,
        output_dir=str(tmp_path / "out"),
        results_root=str(tmp_path / "results"),
        init_checkpoint=str(ckpt),
        energy_gate_mode=None,
        energy_gate_min_scale=None,
        energy_gate_exponent=None,
        terminal_energy_bonus=None,
        terminal_eta_target=None,
        terminal_shaft_ratio_min=None,
        i_soft_limit=None,
        i_soft_penalty=None,
        hidden_sizes_override=None,
    )

    assert seen["candidate_json"] is None
    assert seen["candidate_tags"] == ""


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


@pytest.mark.parametrize("control_mode", ["foc_assist", "ai_speed"])
def test_build_env_enables_two_action_id_control_for_new_modes(tmp_path: Path, control_mode: str) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text("from config.env_demo_true_motor1 import *  # noqa: F401,F403\n", encoding="utf-8")

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode=control_mode,
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

    assert env.cfg.control_mode == control_mode
    assert env.cfg.enable_id_control is True


def test_build_env_propagates_foc_assist_reward_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text(
        "\n".join(
            [
                "from config.env_demo_true_motor1 import *  # noqa: F401,F403",
                "foc_assist_reward_mode = 'energy'",
                "w_foc_speed = 1.7",
                "w_foc_power = 0.8",
                "w_foc_current = 0.2",
                "w_foc_action = 0.03",
                "foc_speed_tol = 0.17",
                "p_el_tau = 0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode="foc_assist",
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

    assert env.cfg.foc_assist_reward_mode == "energy"
    assert env.cfg.w_foc_speed == pytest.approx(1.7)
    assert env.cfg.w_foc_power == pytest.approx(0.8)
    assert env.cfg.w_foc_current == pytest.approx(0.2)
    assert env.cfg.w_foc_action == pytest.approx(0.03)
    assert env.cfg.foc_speed_tol == pytest.approx(0.17)
    assert env.cfg.p_el_tau == pytest.approx(0.04)


def test_build_env_sets_tracking_reward_for_ai_speed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text(
        "\n".join(
            [
                "from config.env_demo_true_motor1 import *  # noqa: F401,F403",
                "baseline_speed_err = 2.5",
                "baseline_current_rms = 1.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode="ai_speed",
        w_speed=1.7,
        w_power=2.0,
        w_current=0.2,
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

    assert env.cfg.w_speed_error == pytest.approx(1.7)
    assert env.cfg.w_current_rms == pytest.approx(0.2)
    assert env.cfg.baseline_speed_err == pytest.approx(2.5)
    assert env.cfg.baseline_current_rms == pytest.approx(1.25)


@pytest.mark.parametrize("control_mode", ["ai_current", "ai_speed", "foc_assist"])
def test_adapt_checkpoint_state_dict_remaps_single_action_id_head_to_second_slot(control_mode: str) -> None:
    source = {
        "actor_mu.weight": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        "actor_mu.bias": torch.tensor([4.0], dtype=torch.float32),
        "log_std": torch.tensor([5.0], dtype=torch.float32),
    }
    target = {
        "actor_mu.weight": torch.zeros((2, 3), dtype=torch.float32),
        "actor_mu.bias": torch.zeros((2,), dtype=torch.float32),
        "log_std": torch.zeros((2,), dtype=torch.float32),
    }

    adapted, adjusted = _adapt_checkpoint_state_dict_for_model(
        source,
        target,
        target_control_mode=control_mode,
    )

    assert "actor_mu.weight" in adjusted
    assert "actor_mu.bias" in adjusted
    assert "log_std" in adjusted
    assert adapted["actor_mu.weight"].tolist() == [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    assert adapted["actor_mu.bias"].tolist() == [0.0, 4.0]
    assert adapted["log_std"].tolist() == [0.0, 5.0]


def test_train_infers_feature_keys_from_init_checkpoint_before_build_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "init_actor.pth"
    inferred_keys = build_feature_keys(True, False)
    agent = PPOVoltageAgent(feature_keys=inferred_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    torch.save(agent.net.state_dict(), ckpt)

    seen: dict[str, object] = {}

    def _fake_build_env(*args, **kwargs):
        seen["feature_keys"] = list(kwargs["feature_keys"])
        raise RuntimeError("stop_after_build_env")

    monkeypatch.setattr("mic_ai.ai.train_ai_id_ref.build_env", _fake_build_env)

    with pytest.raises(RuntimeError, match="stop_after_build_env"):
        train(
            env_config="config/env_demo_true_motor1.py",
            episodes=1,
            episode_steps=1,
            control_mode="ai_id_ref",
            w_speed=1.0,
            w_power=1.0,
            w_current=None,
            w_smooth=0.0,
            w_mag=0.0,
            w_shaft=0.0,
            w_eta=0.0,
            w_eta_episode=0.0,
            eta_clip=1.2,
            id_ref_alpha=1.0,
            id_ref_rate_limit=None,
            ai_id_speed_tol=0.5,
            ai_id_speed_tol_rel=None,
            id_ref_gate_speed_tol=None,
            id_ref_gate_speed_tol_rel=None,
            id_ref_gate_min_scale=0.0,
            id_ref_gate_exponent=1.0,
            fast=True,
            time_budget_min=None,
            override_load_torque=True,
            override_omega_ref=True,
            ai_id_ref_relative=True,
            delta_id_max=0.1,
            load_torque=None,
            omega_ref_override=None,
            scenarios=[],
            scenario_sample="random",
            omega_ref_range=None,
            load_torque_range=None,
            seed=123,
            sigma_start=0.1,
            sigma_end=0.1,
            sigma_decay_episodes=1,
            power_warmup_episodes=0,
            power_ramp_episodes=0,
            energy_warmup_episodes=0,
            energy_ramp_episodes=0,
            eval_interval=0,
            eval_scenarios="speed_step",
            eval_dt=None,
            eval_t_end=None,
            eval_window_frac=0.25,
            eval_error_tol_rel=0.05,
            eval_error_tol_abs=0.0,
            eval_use_total_power=True,
            include_energy_obs=False,
            include_episode_eta_obs=False,
            update_every_episodes=1,
            lr=5e-4,
            entropy_coef=0.005,
            actor_anchor_coef=0.0,
            external_step27_select=False,
            init_checkpoint=str(ckpt),
        )

    assert seen["feature_keys"] == inferred_keys


def test_apply_env_reward_overrides_mutates_env_cfg(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text("from config.env_demo_true_motor1 import *  # noqa: F401,F403\n", encoding="utf-8")

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

    _apply_env_reward_overrides(
        env,
        energy_gate_mode="soft",
        energy_gate_min_scale=0.2,
        energy_gate_exponent=1.3,
        terminal_energy_bonus=0.4,
        terminal_eta_target=0.25,
        terminal_shaft_ratio_min=0.85,
        i_soft_limit=0.95,
        i_soft_penalty=0.6,
    )

    assert env.cfg.ai_id_energy_gate_mode == "soft"
    assert env.cfg.ai_id_energy_gate_min_scale == pytest.approx(0.2)
    assert env.cfg.ai_id_energy_gate_exponent == pytest.approx(1.3)
    assert env.cfg.ai_id_terminal_energy_bonus == pytest.approx(0.4)
    assert env.cfg.ai_id_terminal_eta_target == pytest.approx(0.25)
    assert env.cfg.ai_id_terminal_shaft_ratio_min == pytest.approx(0.85)
    assert env.cfg.i_soft_limit == pytest.approx(0.95)
    assert env.cfg.i_soft_penalty == pytest.approx(0.6)


def test_parse_scenario_reward_overrides_from_json_file(tmp_path: Path) -> None:
    payload = {
        "load_step": {"w_speed": 3.0, "w_power": 4.0, "ai_id_speed_tol_rel": 0.04, "reward_start_frac": 0.3},
        "speed_step": {"w_speed": 2.4, "terminal_energy_bonus": 1.25},
    }
    path = tmp_path / "scenario_overrides.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8-sig")

    parsed = _parse_scenario_reward_overrides(str(path))

    assert parsed is not None
    assert parsed["load_step"]["w_speed"] == pytest.approx(3.0)
    assert parsed["load_step"]["w_power"] == pytest.approx(4.0)
    assert parsed["load_step"]["ai_id_speed_tol_rel"] == pytest.approx(0.04)
    assert parsed["load_step"]["reward_start_frac"] == pytest.approx(0.3)
    assert parsed["speed_step"]["w_speed"] == pytest.approx(2.4)
    assert parsed["speed_step"]["terminal_energy_bonus"] == pytest.approx(1.25)


def test_parse_seed_scenario_reward_overrides_from_json_file(tmp_path: Path) -> None:
    payload = {
        "505": {"start_stop": {"w_speed": 4.0, "id_ref_gate_min_scale": 0.25}},
        "202": {"start_stop": {"w_eta_episode": 1.8, "reward_start_frac": 0.15}},
    }
    path = tmp_path / "seed_scenario_overrides.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8-sig")

    parsed = _parse_seed_scenario_reward_overrides(str(path))

    assert parsed is not None
    assert parsed[505]["start_stop"]["w_speed"] == pytest.approx(4.0)
    assert parsed[505]["start_stop"]["id_ref_gate_min_scale"] == pytest.approx(0.25)
    assert parsed[202]["start_stop"]["w_eta_episode"] == pytest.approx(1.8)
    assert parsed[202]["start_stop"]["reward_start_frac"] == pytest.approx(0.15)


def test_apply_scenario_reward_overrides_mutates_id_ref_weights(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text("from config.env_demo_true_motor1 import *  # noqa: F401,F403\n", encoding="utf-8")

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode="ai_id_ref",
        w_speed=1.0,
        w_power=2.0,
        w_current=None,
        w_smooth=0.05,
        w_mag=0.0,
        w_shaft=0.5,
        w_eta=0.3,
        w_eta_episode=0.1,
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

    eff = _apply_scenario_reward_overrides(
        env,
        scenario_name="load_step",
        episode_seed=None,
        base_w_speed=1.0,
        base_w_power=2.0,
        base_w_shaft=0.5,
        base_w_eta=0.3,
        base_w_eta_episode=0.1,
        base_reward_start_frac=0.0,
        base_terminal_energy_bonus=0.8,
        base_ai_id_speed_tol=0.5,
        base_ai_id_speed_tol_rel=None,
        base_id_ref_gate_speed_tol=None,
        base_id_ref_gate_speed_tol_rel=None,
        base_id_ref_gate_min_scale=0.0,
        base_id_ref_gate_exponent=1.0,
        scenario_reward_overrides={
            "load_step": {
                "w_speed": 3.0,
                "w_power": 4.0,
                "reward_start_frac": 0.35,
                "terminal_energy_bonus": 1.6,
                "ai_id_speed_tol": 0.35,
                "ai_id_speed_tol_rel": 0.04,
                "id_ref_gate_speed_tol_rel": 0.08,
                "id_ref_gate_min_scale": 0.2,
                "id_ref_gate_exponent": 1.4,
            }
        },
        seed_scenario_reward_overrides=None,
    )

    assert eff["w_speed"] == pytest.approx(3.0)
    assert eff["w_power"] == pytest.approx(4.0)
    assert eff["w_shaft"] == pytest.approx(0.5)
    assert eff["w_eta"] == pytest.approx(0.3)
    assert eff["w_eta_episode"] == pytest.approx(0.1)
    assert eff["reward_start_frac"] == pytest.approx(0.35)
    assert eff["terminal_energy_bonus"] == pytest.approx(1.6)
    assert eff["ai_id_speed_tol"] == pytest.approx(0.35)
    assert eff["ai_id_speed_tol_rel"] == pytest.approx(0.04)
    assert eff["id_ref_gate_speed_tol"] is None
    assert eff["id_ref_gate_speed_tol_rel"] == pytest.approx(0.08)
    assert eff["id_ref_gate_min_scale"] == pytest.approx(0.2)
    assert eff["id_ref_gate_exponent"] == pytest.approx(1.4)
    assert env.cfg.w_ai_id_speed == pytest.approx(3.0)
    assert env.cfg.w_ai_id_power == pytest.approx(4.0)
    assert env.cfg.w_ai_id_shaft == pytest.approx(0.5)
    assert env.cfg.w_ai_id_eta == pytest.approx(0.3)
    assert env.cfg.w_ai_id_eta_episode == pytest.approx(0.1)
    assert env.cfg.ai_id_terminal_energy_bonus == pytest.approx(1.6)
    assert env.cfg.ai_id_speed_tol == pytest.approx(0.35)
    assert env.cfg.ai_id_speed_tol_rel == pytest.approx(0.04)
    assert env.cfg.id_ref_gate_speed_tol is None
    assert env.cfg.id_ref_gate_speed_tol_rel == pytest.approx(0.08)
    assert env.cfg.id_ref_gate_min_scale == pytest.approx(0.2)
    assert env.cfg.id_ref_gate_exponent == pytest.approx(1.4)


def test_apply_scenario_reward_overrides_allows_seed_specific_override_to_win(tmp_path: Path) -> None:
    cfg_path = tmp_path / "env_cfg.py"
    cfg_path.write_text("from config.env_demo_true_motor1 import *  # noqa: F401,F403\n", encoding="utf-8")

    env = build_env(
        str(cfg_path),
        episode_steps=5,
        control_mode="ai_id_ref",
        w_speed=1.0,
        w_power=2.0,
        w_current=None,
        w_smooth=0.05,
        w_mag=0.0,
        w_shaft=0.5,
        w_eta=0.3,
        w_eta_episode=0.1,
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

    eff = _apply_scenario_reward_overrides(
        env,
        scenario_name="start_stop",
        episode_seed=505,
        base_w_speed=1.0,
        base_w_power=2.0,
        base_w_shaft=0.5,
        base_w_eta=0.3,
        base_w_eta_episode=0.1,
        base_reward_start_frac=0.0,
        base_terminal_energy_bonus=0.8,
        base_ai_id_speed_tol=0.5,
        base_ai_id_speed_tol_rel=None,
        base_id_ref_gate_speed_tol=None,
        base_id_ref_gate_speed_tol_rel=None,
        base_id_ref_gate_min_scale=0.0,
        base_id_ref_gate_exponent=1.0,
        scenario_reward_overrides={
            "start_stop": {
                "w_speed": 3.0,
                "w_eta_episode": 0.9,
                "id_ref_gate_min_scale": 0.12,
            }
        },
        seed_scenario_reward_overrides={
            505: {
                "start_stop": {
                    "w_speed": 4.2,
                    "ai_id_speed_tol_rel": 0.03,
                    "id_ref_gate_min_scale": 0.22,
                }
            }
        },
    )

    assert eff["w_speed"] == pytest.approx(4.2)
    assert eff["w_eta_episode"] == pytest.approx(0.9)
    assert eff["ai_id_speed_tol_rel"] == pytest.approx(0.03)
    assert eff["id_ref_gate_min_scale"] == pytest.approx(0.22)
    assert env.cfg.w_ai_id_speed == pytest.approx(4.2)
    assert env.cfg.ai_id_speed_tol_rel == pytest.approx(0.03)
    assert env.cfg.id_ref_gate_min_scale == pytest.approx(0.22)


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


def test_adapt_checkpoint_state_dict_for_model_zero_pads_expanded_hidden_layers() -> None:
    state = {
        "actor_body.0.weight": torch.ones(64, 11),
        "actor_body.0.bias": torch.ones(64),
        "actor_body.2.weight": torch.ones(64, 64),
        "actor_body.2.bias": torch.ones(64),
    }
    model_state = {
        "actor_body.0.weight": torch.zeros(96, 11),
        "actor_body.0.bias": torch.zeros(96),
        "actor_body.2.weight": torch.zeros(96, 96),
        "actor_body.2.bias": torch.zeros(96),
    }

    adapted, adjusted = _adapt_checkpoint_state_dict_for_model(state, model_state)

    assert set(adjusted) == set(state.keys())
    assert adapted["actor_body.0.weight"].shape == (96, 11)
    assert torch.equal(adapted["actor_body.0.weight"][:64, :], torch.ones(64, 11))
    assert torch.equal(adapted["actor_body.0.weight"][64:, :], torch.zeros(32, 11))
    assert adapted["actor_body.2.weight"].shape == (96, 96)
    assert torch.equal(adapted["actor_body.2.weight"][:64, :64], torch.ones(64, 64))
    assert torch.equal(adapted["actor_body.2.weight"][64:, :], torch.zeros(32, 96))
    assert torch.equal(adapted["actor_body.2.weight"][:, 64:], torch.zeros(96, 32))
    assert torch.equal(adapted["actor_body.2.bias"][:64], torch.ones(64))
    assert torch.equal(adapted["actor_body.2.bias"][64:], torch.zeros(32))


def test_curriculum_scale_handles_warmup_and_ramp() -> None:
    assert _curriculum_scale(episode=0, warmup_episodes=0, ramp_episodes=0) == pytest.approx(1.0)
    assert _curriculum_scale(episode=0, warmup_episodes=3, ramp_episodes=4) == pytest.approx(0.0)
    assert _curriculum_scale(episode=2, warmup_episodes=3, ramp_episodes=4) == pytest.approx(0.0)
    assert _curriculum_scale(episode=3, warmup_episodes=3, ramp_episodes=4) == pytest.approx(0.0)
    assert _curriculum_scale(episode=5, warmup_episodes=3, ramp_episodes=4) == pytest.approx(0.5)
    assert _curriculum_scale(episode=7, warmup_episodes=3, ramp_episodes=4) == pytest.approx(1.0)


def test_infer_hidden_sizes_from_state_dict_reads_actor_body_layers() -> None:
    state = {
        "actor_body.0.weight": torch.zeros(64, 11),
        "actor_body.0.bias": torch.zeros(64),
        "actor_body.2.weight": torch.zeros(32, 64),
        "critic_body.0.weight": torch.zeros(64, 11),
    }

    assert _infer_hidden_sizes_from_state_dict(state) == (64, 32)


def test_parse_hidden_sizes_accepts_csv() -> None:
    assert _parse_hidden_sizes("96, 128") == (96, 128)
