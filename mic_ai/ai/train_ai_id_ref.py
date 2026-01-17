from __future__ import annotations

import sys
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


FEATURE_KEYS = [
    "omega_norm",
    "omega_ref_norm",
    "err_norm",
    "id_norm",
    "iq_norm",
    "slip_norm",
    "load_torque_norm",
]

OUTPUT_DIR = Path("outputs/ai_id_ref")
EPISODE_LOG_DIR = OUTPUT_DIR / "episode_logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = Path("results_run")


def _parse_scenarios(text: str) -> List[str]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    return names


def _prepare_output_file(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        path.rename(backup)
        print(f"[log] existing {path.name} -> backup {backup.name}")
    return path


def build_env(
    env_config_path: str,
    episode_steps: int,
    w_speed: float,
    w_power: float,
    w_smooth: float,
    w_mag: float,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    load_torque: float | None,
    omega_ref_override: float | None,
) -> MicAiAIEnv:
    env_sim = make_env_from_config(env_config_path)
    env_cfg = env_sim.env_config

    if omega_ref_override is None:
        omega_ref = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
    else:
        omega_ref = float(omega_ref_override)
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    i_limit = float(max(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base * 8.0), i_base * 8.0, 5.0))
    id_ref_base = float(getattr(getattr(env_cfg, "foc", None), "id_ref", 0.0) or 0.0)
    id_ref_max = max(i_base * 1.5, id_ref_base, id_ref_base * 1.2)

    cfg = load_ai_voltage_config()
    curriculum_cfg = get_curriculum_config(cfg)
    piecewise_steps = curriculum_cfg.get("piecewise_steps", (150, 300))
    piecewise_multipliers = curriculum_cfg.get("piecewise_multipliers", (1.0, 0.8, 1.0))
    curriculum_stages = curriculum_cfg.get("omega_pu_stages", (0.3, 0.5))
    stage_boundaries = curriculum_cfg.get("stage_episode_boundaries", (150, 300))

    ai_cfg = AiEnvConfig(
        episode_steps=int(episode_steps),
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        omega_ref_max=max(abs(omega_ref) * 1.2, 1e-3),
        w_speed_error=0.0,
        w_current_rms=0.0,
        i_base=i_base,
        i_max=i_limit,
        control_mode="ai_id_ref",
        reward_min=-10.0,
        reward_max=1.0,
        w_ai_id_speed=float(w_speed),
        w_ai_id_power=float(w_power),
        w_ai_id_smooth=float(w_smooth),
        w_ai_id_mag=float(w_mag),
        id_ref_alpha=float(id_ref_alpha),
        id_ref_rate_limit=None if id_ref_rate_limit is None else float(id_ref_rate_limit),
        id_ref_gate_speed_tol=None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(id_ref_gate_min_scale),
        id_ref_gate_exponent=float(id_ref_gate_exponent),
        delta_id_max=float(delta_id_max),
        ai_id_speed_tol=float(ai_id_speed_tol),
        ai_id_speed_tol_rel=None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
        curriculum_omega_pu=tuple(float(x) for x in curriculum_stages),
        curriculum_stage_episodes=tuple(int(x) for x in stage_boundaries),
        omega_piecewise_steps=tuple(int(x) for x in piecewise_steps),
        omega_piecewise_multipliers=tuple(float(x) for x in piecewise_multipliers),
        id_ref_min=0.0,
        id_ref_max=float(id_ref_max),
        ai_id_ref_relative=bool(ai_id_ref_relative),
        i_hard_limit=float(i_limit * 2.0),
        load_torque_override=None if load_torque is None else float(load_torque),
        override_load_torque=bool(override_load_torque),
        override_omega_ref=bool(override_omega_ref),
    )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    if load_torque is None:
        base_env.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)
    else:
        base_env.load_torque_func = lambda _t, load=load_torque: float(load)

    return MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None, world_input_keys=FEATURE_KEYS, world_target_keys=["omega_norm"])


def train(
    env_config: str,
    episodes: int,
    episode_steps: int,
    w_speed: float,
    w_power: float,
    w_smooth: float,
    w_mag: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    fast: bool,
    time_budget_min: float | None,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    load_torque: float | None,
    omega_ref_override: float | None,
    scenarios: List[str] | None,
    scenario_sample: str,
) -> Dict[str, str]:
    env = build_env(
        env_config,
        episode_steps=episode_steps,
        w_speed=w_speed,
        w_power=w_power,
        w_smooth=w_smooth,
        w_mag=w_mag,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=ai_id_ref_relative,
        delta_id_max=delta_id_max,
        id_ref_alpha=id_ref_alpha,
        id_ref_rate_limit=id_ref_rate_limit,
        ai_id_speed_tol=ai_id_speed_tol,
        ai_id_speed_tol_rel=ai_id_speed_tol_rel,
        id_ref_gate_speed_tol=id_ref_gate_speed_tol,
        id_ref_gate_speed_tol_rel=id_ref_gate_speed_tol_rel,
        id_ref_gate_min_scale=id_ref_gate_min_scale,
        id_ref_gate_exponent=id_ref_gate_exponent,
        load_torque=load_torque,
        omega_ref_override=omega_ref_override,
    )

    scenarios = [s for s in (scenarios or []) if s]
    scenario_sample = str(scenario_sample or "random").lower()
    rng = np.random.default_rng()

    hidden_sizes = (64, 64) if fast else (128, 128)
    train_epochs = 3 if fast else 5
    minibatch_frac = 0.5 if fast else 0.25
    agent = PPOVoltageAgent(
        feature_keys=FEATURE_KEYS,
        action_dim=1,
        device="cpu",
        hidden_sizes=hidden_sizes,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.005,
        value_coef=0.3,
        max_grad_norm=0.5,
        train_epochs=train_epochs,
        minibatch_frac=minibatch_frac,
    )

    env_name = Path(env_config).stem
    ckpt_dir = (CHECKPOINT_DIR / env_name).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episodes_log: List[Dict[str, float]] = []
    best_score = float("inf")
    best_ckpt: Path | None = None
    t0 = time.perf_counter()
    max_seconds = None if time_budget_min is None else float(time_budget_min) * 60.0

    for ep in range(int(episodes)):
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            print(f"[{env_name}] time budget reached at ep {ep}")
            break

        scenario_name = ""
        if scenarios:
            if scenario_sample == "cycle":
                scenario_name = scenarios[ep % len(scenarios)]
            else:
                scenario_name = str(rng.choice(scenarios))
            env.set_scenario(scenario_name)

        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        agent.set_action_std(0.2 if ep < 50 else 0.05)

        while not done and steps < int(episode_steps):
            action, logp, value = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            agent.store(obs, action, logp, reward, done, value)
            total_reward += float(reward)
            obs = obs_next
            steps += 1

        with torch.no_grad():
            last_value = float(agent.net(agent._to_tensor(obs).unsqueeze(0))[2].item())
        losses = agent.update(last_value=last_value)
        m = env.episode_metrics()

        entry = {
            "episode": float(ep),
            "steps": float(m.get("steps", steps)),
            "mean_speed_error": float(m.get("mean_speed_error", 0.0)),
            "mean_p_in_pos": float(m.get("mean_p_in_pos", 0.0)),
            "mean_current_rms": float(m.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(m.get("action_norm", 0.0)),
            "mean_reward": float(total_reward / max(steps, 1)),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "value_loss": float(losses.get("value_loss", 0.0)),
            "scenario": scenario_name,
        }
        episodes_log.append(entry)

        # Score: prioritize power, but keep speed error bounded.
        score = entry["mean_p_in_pos"] + 50.0 * entry["mean_speed_error"]
        if score < best_score:
            best_score = score
            best_ckpt = ckpt_dir / "best_actor.pth"
            torch.save(agent.net.state_dict(), best_ckpt)

        if ep % 10 == 0 or ep == episodes - 1:
            print(
                f"[{env_name}] ep {ep:03d} | mean_p_in_pos {entry['mean_p_in_pos']:.3f} | "
                f"mean|e_w| {entry['mean_speed_error']:.3f} | act_norm {entry['mean_action_norm']:.3f}"
            )

    last_ckpt = ckpt_dir / "last_actor.pth"
    torch.save(agent.net.state_dict(), last_ckpt)

    episodes_path = _prepare_output_file(EPISODE_LOG_DIR / f"ai_id_ref_{env_name}_episodes.json")
    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)

    run_dir = RESULTS_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{env_name}_ai_id_ref"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)
    torch.save(agent.net.state_dict(), run_dir / "actor_critic.pth")

    if best_ckpt is None:
        best_ckpt = ckpt_dir / "best_actor.pth"
        shutil.copyfile(last_ckpt, best_ckpt)

    print(f"Saved checkpoints: {best_ckpt} | {last_ckpt}")
    return {"episodes": str(episodes_path), "best": str(best_ckpt), "last": str(last_ckpt), "run_dir": str(run_dir)}


def main() -> None:
    p = argparse.ArgumentParser(description="Train AI to adapt FOC id_ref for efficiency (minimize P_in).")
    p.add_argument("config", help="Env config path (.py)")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--w-speed", type=float, default=1.0)
    p.add_argument("--w-power", type=float, default=6.0)
    p.add_argument("--w-smooth", type=float, default=0.05)
    p.add_argument("--w-mag", type=float, default=0.0)
    p.add_argument("--ai-id-speed-tol", type=float, default=0.5)
    p.add_argument("--ai-id-speed-tol-rel", type=float, default=None, help="Relative speed tol (e.g., 0.05).")
    p.add_argument("--id-ref-alpha", type=float, default=1.0)
    p.add_argument("--id-ref-rate-limit", type=float, default=None, help="Max d(id_ref)/dt, A/s.")
    p.add_argument("--id-ref-gate-speed-tol", type=float, default=None, help="Gate id_ref when |e_omega| exceeds tol.")
    p.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=None, help="Relative gate tol (e.g., 0.05).")
    p.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    p.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=None)
    p.add_argument("--override-load-torque", action="store_true", help="Force zero load during training.")
    p.add_argument("--no-override-omega-ref", dest="override_omega_ref", action="store_false", help="Use scenario omega_ref.")
    p.add_argument("--relative", action="store_true", help="Interpret action as delta around base id_ref.")
    p.add_argument("--delta-id-max", type=float, default=0.3, help="Relative id_ref delta scale.")
    p.add_argument("--load-torque", type=float, default=None, help="Override constant load torque, N*m.")
    p.add_argument("--omega-ref", type=float, default=None, help="Override omega_ref, rad/s.")
    p.add_argument("--omega-ref-pu", type=float, default=0.8, help="Omega_ref as pu of base omega (2*pi*10/p).")
    p.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario list (e.g., speed_step,ramp,load_step,start_stop).")
    p.add_argument("--scenario-sample", type=str, default="random", choices=["random", "cycle"])
    p.set_defaults(override_omega_ref=True)
    args = p.parse_args()
    omega_ref_override = None
    if args.omega_ref is not None:
        omega_ref_override = float(args.omega_ref)
    elif args.omega_ref_pu is not None:
        env_cfg = make_env_from_config(args.config).env_config
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_override = float(args.omega_ref_pu) * omega_base

    scenarios = _parse_scenarios(args.scenarios)
    override_omega_ref = bool(args.override_omega_ref)
    override_load_torque = bool(args.override_load_torque)
    if scenarios:
        override_omega_ref = False
        override_load_torque = False

    train(
        env_config=args.config,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        w_speed=args.w_speed,
        w_power=args.w_power,
        w_smooth=args.w_smooth,
        w_mag=args.w_mag,
        id_ref_alpha=float(args.id_ref_alpha),
        id_ref_rate_limit=None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
        ai_id_speed_tol=float(args.ai_id_speed_tol),
        ai_id_speed_tol_rel=None if args.ai_id_speed_tol_rel is None else float(args.ai_id_speed_tol_rel),
        id_ref_gate_speed_tol=None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(args.id_ref_gate_min_scale),
        id_ref_gate_exponent=float(args.id_ref_gate_exponent),
        fast=bool(args.fast),
        time_budget_min=args.time_budget_min,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=bool(args.relative),
        delta_id_max=float(args.delta_id_max),
        load_torque=None if args.load_torque is None else float(args.load_torque),
        omega_ref_override=omega_ref_override,
        scenarios=scenarios,
        scenario_sample=str(args.scenario_sample),
    )


if __name__ == "__main__":
    main()
