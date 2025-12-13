from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS, build_env, resolve_config_path, _motor_key_from_config
from mic_ai.ai.ai_voltage_config import (
    get_curriculum_config,
    get_reward_weights,
    get_voltage_scale,
    load_ai_voltage_config,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an ai_voltage PPO checkpoint and save episode metrics to JSON.")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py", help="Env config path (.py).")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint (actor_critic).")
    p.add_argument("--episodes", type=int, default=10, help="Number of eval episodes.")
    p.add_argument("--episode-steps", type=int, default=200, help="Episode steps per eval episode.")
    p.add_argument(
        "--hidden-sizes",
        type=str,
        default=None,
        help="Override hidden sizes, e.g. '64,64'. If omitted, inferred from checkpoint when possible.",
    )
    p.add_argument("--output", default="outputs/demo_ai/episode_logs/ai_voltage_eval_motor1.json", help="Where to save JSON.")
    return p.parse_args(argv)


def _as_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except Exception:
        return float(default)

def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    if not hasattr(w0, "shape") or not hasattr(w2, "shape"):
        return None
    try:
        h1 = int(w0.shape[0])
        h2 = int(w2.shape[0])
    except Exception:
        return None
    if h1 <= 0 or h2 <= 0:
        return None
    return (h1, h2)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_ai_voltage_config()
    env_path = resolve_config_path(str(args.env_config))
    motor_key = _motor_key_from_config(str(env_path))
    reward_weights = get_reward_weights(cfg, motor_key)
    curriculum_cfg = get_curriculum_config(cfg)
    voltage_scale = get_voltage_scale(cfg, motor_key)

    env = build_env(
        env_path,
        episode_steps=int(args.episode_steps),
        reward_weights=reward_weights,
        curriculum_cfg=curriculum_cfg,
        voltage_scale=voltage_scale,
        motor_key=motor_key,
        ident_path=None,
    )

    state = torch.load(Path(args.checkpoint), map_location="cpu")
    hidden_sizes: tuple[int, ...] | None = None
    if args.hidden_sizes:
        hidden_sizes = tuple(int(x.strip()) for x in str(args.hidden_sizes).split(",") if x.strip())
    else:
        hidden_sizes = _infer_hidden_sizes(state)
    agent = PPOVoltageAgent(feature_keys=FEATURE_KEYS, action_dim=2, device="cpu", hidden_sizes=hidden_sizes or (128, 128))
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)

    logs: List[Dict[str, float]] = []
    for ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        step_counter = 0
        while not done and step_counter < int(args.episode_steps):
            action, _logp, _value = agent.act(obs)
            obs, _reward, done, _info = env.step(action)
            step_counter += 1

        m = env.episode_metrics()
        logs.append(
            {
                "episode": float(ep),
                "steps": float(m.get("steps", step_counter)),
                "hard_terminated": float(m.get("hard_terminated", 0.0)),
                "mean_speed_error": _as_float(m.get("mean_speed_error", 0.0)),
                "mean_current_rms": _as_float(m.get("mean_current_rms", 0.0)),
                "mean_p_in": _as_float(m.get("mean_p_in", 0.0)),
                "mean_p_in_pos": _as_float(m.get("mean_p_in_pos", 0.0)),
                "mean_action_norm": _as_float(m.get("action_norm", 0.0)),
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    p_in_pos = np.array([row["mean_p_in_pos"] for row in logs], dtype=float)
    speed_err = np.array([row["mean_speed_error"] for row in logs], dtype=float)
    print(f"Saved eval log to {out_path}")
    print(f"mean_p_in_pos: mean={float(p_in_pos.mean()):.3f} p90={float(np.percentile(p_in_pos, 90)):.3f}")
    print(f"mean_speed_error: mean={float(speed_err.mean()):.3f} p90={float(np.percentile(speed_err, 90)):.3f}")


if __name__ == "__main__":
    main()
