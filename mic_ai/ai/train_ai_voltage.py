from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.simple_agent import ActorCriticAgent
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


FEATURE_KEYS = ["omega_norm", "omega_ref_norm", "err_norm", "id_norm", "iq_norm"]
WORLD_INPUT_KEYS = FEATURE_KEYS + ["action_vd_norm", "action_vq_norm"]
WORLD_TARGET_KEYS = ["omega_norm", "id_norm", "iq_norm"]


def build_env(env_config_path: str, episode_steps: int | None = None) -> MicAiAIEnv:
    env_sim = make_env_from_config(env_config_path)
    env_cfg = env_sim.env_config
    omega_ref = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))  # немного ниже для лучшей обучаемости
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    steps = episode_steps or int(max(env_cfg.sim.t_end / env_cfg.sim.dt, 1))

    ai_cfg = AiEnvConfig(
        episode_steps=steps,
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        w_speed_error=0.1,
        w_current_rms=0.01,
        i_base=i_base,
        control_mode="ai_voltage",
        v_max=None,
        reward_clip=5.0,
        w_int_scale=0.0,
        curiosity_beta=0.0,
        wm_lr=1e-4,
    )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    base_env.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)

    env = MicAiAIEnv(
        base_env,
        ai_cfg,
        curiosity=None,
        world_model=None,
        world_input_keys=WORLD_INPUT_KEYS,
        world_target_keys=WORLD_TARGET_KEYS,
    )
    return env


def train(env_config: str, episodes: int, episode_steps: int | None, output_prefix: str) -> Dict[str, str]:
    env = build_env(env_config, episode_steps)
    agent = ActorCriticAgent(
        feature_keys=FEATURE_KEYS,
        action_dim=2,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        sigma=0.2,
        max_grad_norm=5.0,
    )

    env_name = Path(env_config).stem
    out_dir = Path(output_prefix).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_log: List[Dict[str, object]] = []
    learning_log: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs = env.reset()
        agent.start_episode()
        done = False
        traj_t: List[float] = []
        traj_omega: List[float] = []
        traj_omega_ref: List[float] = []
        traj_reward: List[float] = []
        traj_vd: List[float] = []
        traj_vq: List[float] = []
        traj_speed_err: List[float] = []
        traj_current_rms: List[float] = []

        total_reward = 0.0
        while not done:
            action = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            agent.record_reward(reward, next_obs=obs_next)
            total_reward += float(reward)

            traj_t.append(info.get("t", len(traj_t) * env.cfg.dt))
            traj_omega.append(info.get("omega_meas", 0.0))
            traj_omega_ref.append(info.get("omega_ref", env.cfg.omega_ref))
            traj_reward.append(float(reward))
            traj_vd.append(float(info.get("action_vd_norm", 0.0)))
            traj_vq.append(float(info.get("action_vq_norm", 0.0)))
            traj_speed_err.append(abs(float(info.get("speed_err", 0.0))))
            traj_current_rms.append(float(info.get("i_rms", 0.0)))
            obs = obs_next

        losses = agent.update_after_episode()
        metrics = env.episode_metrics()
        metrics.update(
            {
                "episode": ep,
                "total_reward": total_reward,
                "actor_loss": float(losses.get("actor_loss", 0.0)),
                "critic_loss": float(losses.get("critic_loss", 0.0)),
            }
        )
        learning_log.append(metrics)

        episodes_log.append(
            {
                "episode_idx": ep,
                "t": traj_t,
                "omega": traj_omega,
                "omega_ref": traj_omega_ref,
                "reward": traj_reward,
                "v_d_norm": traj_vd,
                "v_q_norm": traj_vq,
                "speed_err": traj_speed_err,
                "current_rms": traj_current_rms,
                "total_reward": total_reward,
                "mean_speed_error": float(np.mean(traj_speed_err)) if traj_speed_err else 0.0,
                "mean_current_rms": float(np.mean(traj_current_rms)) if traj_current_rms else 0.0,
            }
        )

        print(
            f"[{env_name}] ep {ep:03d} | total_r {total_reward:8.3f} | "
            f"mean|e_w| {metrics['mean_speed_error']:.4f} | mean_i_rms {metrics['mean_current_rms']:.4f}"
        )

    episodes_path = out_dir / f"ai_voltage_{env_name}_episodes.json"
    learning_path = out_dir / f"ai_voltage_learning_{env_name}.json"
    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "env_config": env_config,
                "episodes": episodes_log,
                "ai_env_cfg": env.cfg.__dict__,
            },
            f,
            indent=2,
        )
    with learning_path.open("w", encoding="utf-8") as f:
        json.dump(learning_log, f, indent=2)

    return {"episodes": str(episodes_path), "learning": str(learning_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI agent that drives dq voltages directly.")
    parser.add_argument("--env-configs", nargs="+", required=True, help="Paths to motor config modules.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes.")
    parser.add_argument("--episode-steps", type=int, default=None, help="Override steps per episode.")
    parser.add_argument("--output-prefix", type=str, default="outputs/demo_ai", help="Output folder prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for cfg in args.env_configs:
        paths = train(cfg, episodes=args.episodes, episode_steps=args.episode_steps, output_prefix=args.output_prefix)
        print(f"Saved: {paths['episodes']} | {paths['learning']}")


if __name__ == "__main__":
    main()
