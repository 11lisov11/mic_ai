from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import NAMEPLATE_N_RATED, NAMEPLATE_P_KW
from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
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
    "prev_delta_norm",
    "prev_delta_id_norm",
]

OUTPUT_DIR = Path("outputs/ai_foc_assist")
EPISODE_LOG_DIR = OUTPUT_DIR / "episode_logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = Path("results_run")


def _prepare_output_file(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        path.rename(backup)
        print(f"[log] existing {path.name} -> backup {backup.name}")
    return path


def _rated_omega() -> float:
    return float(2.0 * math.pi * NAMEPLATE_N_RATED / 60.0)


def _rated_torque() -> float:
    omega = _rated_omega()
    return float(NAMEPLATE_P_KW * 1000.0 / max(omega, 1e-6))


def _parse_csv_floats(text: str) -> List[float]:
    items = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            items.append(float(part))
    return items


def build_env(
    env_config_path: str,
    episode_steps: int,
    omega_ref: float,
    w_speed: float,
    w_power: float,
    w_current: float,
    w_action: float,
    speed_tol: float,
    p_el_tau: float,
    delta_iq_max: float,
    delta_id_max: float,
) -> MicAiAIEnv:
    env_sim = make_env_from_config(env_config_path)
    env_cfg = env_sim.env_config

    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    iq_limit = float(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base * 8.0))
    i_limit = max(iq_limit, i_base)

    ai_cfg = AiEnvConfig(
        episode_steps=int(episode_steps),
        dt=float(env_cfg.sim.dt),
        omega_ref=float(omega_ref),
        omega_ref_max=max(abs(omega_ref), 1e-3),
        w_speed_error=0.0,
        w_current_rms=0.0,
        i_base=i_base,
        i_max=i_limit,
        i_hard_limit=float(i_limit * 1.2),
        control_mode="foc_assist",
        enable_id_control=True,
        delta_iq_max=float(delta_iq_max),
        delta_id_max=float(delta_id_max),
        reward_min=-10.0,
        reward_max=1.0,
        foc_assist_reward_mode="energy",
        w_foc_speed=float(w_speed),
        w_foc_power=float(w_power),
        w_foc_current=float(w_current),
        w_foc_action=float(w_action),
        foc_speed_tol=float(speed_tol),
        p_el_tau=float(p_el_tau),
        curriculum_omega_pu=(1.0,),
        curriculum_stage_episodes=(),
        omega_piecewise_steps=(),
        omega_piecewise_multipliers=(1.0,),
        override_load_torque=False,
    )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    base_env.load_torque_func = lambda _t: float(getattr(env_cfg.sim, "load_torque", 0.0))

    env = MicAiAIEnv(
        base_env,
        ai_cfg,
        curiosity=None,
        world_model=None,
        world_input_keys=[],
        world_target_keys=[],
    )
    return env


def train(
    env_config: str,
    episodes: int,
    episode_steps: int,
    omega_ref: float,
    w_speed: float,
    w_power: float,
    w_current: float,
    w_action: float,
    speed_tol: float,
    p_el_tau: float,
    delta_iq_max: float,
    delta_id_max: float,
    load_values: List[float],
    warmup_episodes: int,
    fast: bool,
    time_budget_min: float | None,
) -> Dict[str, str]:
    env = build_env(
        env_config,
        episode_steps=episode_steps,
        omega_ref=omega_ref,
        w_speed=w_speed,
        w_power=w_power,
        w_current=w_current,
        w_action=w_action,
        speed_tol=speed_tol,
        p_el_tau=p_el_tau,
        delta_iq_max=delta_iq_max,
        delta_id_max=delta_id_max,
    )

    hidden_sizes = (64, 64) if fast else (128, 128)
    train_epochs = 3 if fast else 5
    minibatch_frac = 0.5 if fast else 0.25
    agent = PPOVoltageAgent(
        feature_keys=FEATURE_KEYS,
        action_dim=2,
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

        if load_values:
            load = float(load_values[ep % len(load_values)])
            env.base_env.load_torque_func = lambda _t, load=load: load

        if warmup_episodes > 0 and ep < warmup_episodes:
            env.cfg.w_foc_power = 0.0
        else:
            env.cfg.w_foc_power = float(w_power)

        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        agent.set_action_std(0.15 if ep < 50 else 0.05)

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
        }
        episodes_log.append(entry)

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

    episodes_path = _prepare_output_file(EPISODE_LOG_DIR / f"ai_foc_assist_{env_name}_episodes.json")
    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)

    run_dir = RESULTS_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{env_name}_ai_foc_assist"
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
    p = argparse.ArgumentParser(description="Train MIC AI (FOC assist) for energy efficiency.")
    p.add_argument("config", help="Env config path (.py)")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--omega-ref", type=float, default=None, help="Absolute omega_ref, rad/s.")
    p.add_argument("--omega-ref-pu", type=float, default=0.8, help="Omega_ref as pu of base omega.")
    p.add_argument("--w-speed", type=float, default=5.0)
    p.add_argument("--w-power", type=float, default=3.0)
    p.add_argument("--w-current", type=float, default=0.2)
    p.add_argument("--w-action", type=float, default=0.02)
    p.add_argument("--speed-tol", type=float, default=0.5)
    p.add_argument("--p-el-tau", type=float, default=0.02)
    p.add_argument("--delta-iq-max", type=float, default=0.15)
    p.add_argument("--delta-id-max", type=float, default=0.25)
    p.add_argument("--load-max-pu", type=float, default=0.3)
    p.add_argument("--load-points", type=int, default=6)
    p.add_argument("--warmup-episodes", type=int, default=50)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=None)
    args = p.parse_args()

    env_cfg = make_env_from_config(args.config).env_config
    omega_base = float(2.0 * math.pi * 10.0 / max(env_cfg.motor.p, 1))
    omega_ref = float(args.omega_ref) if args.omega_ref is not None else float(args.omega_ref_pu) * omega_base

    m_nom = _rated_torque()
    load_max = max(0.0, float(args.load_max_pu)) * m_nom
    load_points = max(int(args.load_points), 1)
    load_values = np.linspace(0.0, load_max, load_points).tolist()

    train(
        env_config=args.config,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        omega_ref=omega_ref,
        w_speed=args.w_speed,
        w_power=args.w_power,
        w_current=args.w_current,
        w_action=args.w_action,
        speed_tol=args.speed_tol,
        p_el_tau=args.p_el_tau,
        delta_iq_max=args.delta_iq_max,
        delta_id_max=args.delta_id_max,
        load_values=load_values,
        warmup_episodes=int(args.warmup_episodes),
        fast=bool(args.fast),
        time_budget_min=args.time_budget_min,
    )


if __name__ == "__main__":
    main()
