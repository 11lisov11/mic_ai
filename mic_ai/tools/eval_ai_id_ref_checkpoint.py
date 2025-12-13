from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS, build_env


def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    try:
        return (int(w0.shape[0]), int(w2.shape[0]))
    except Exception:
        return None


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an ai_id_ref PPO checkpoint and save episode metrics to JSON.")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--output", default="outputs/ai_id_ref/episode_logs/ai_id_ref_eval.json")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    env = build_env(str(args.env_config), episode_steps=int(args.episode_steps), w_speed=1.0, w_power=6.0, w_smooth=0.05)
    state = torch.load(Path(args.checkpoint), map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)
    agent = PPOVoltageAgent(feature_keys=FEATURE_KEYS, action_dim=1, device="cpu", hidden_sizes=hidden)
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)

    logs: List[Dict[str, float]] = []
    for ep in range(int(args.episodes)):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < int(args.episode_steps):
            action, _lp, _v = agent.act(obs)
            obs, _r, done, _info = env.step(action)
            steps += 1
        m = env.episode_metrics()
        logs.append(
            {
                "episode": float(ep),
                "steps": float(m.get("steps", steps)),
                "mean_speed_error": float(m.get("mean_speed_error", 0.0)),
                "mean_p_in_pos": float(m.get("mean_p_in_pos", 0.0)),
                "mean_current_rms": float(m.get("mean_current_rms", 0.0)),
                "mean_action_norm": float(m.get("action_norm", 0.0)),
                "hard_terminated": float(m.get("hard_terminated", 0.0)),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(logs, indent=2), encoding="utf-8")

    p = np.array([e["mean_p_in_pos"] for e in logs], dtype=float)
    s = np.array([e["mean_speed_error"] for e in logs], dtype=float)
    print(f"Saved eval log to {out}")
    print(f"mean_p_in_pos: mean={float(p.mean()):.3f} p90={float(np.percentile(p,90)):.3f}")
    print(f"mean_speed_error: mean={float(s.mean()):.3f} p90={float(np.percentile(s,90)):.3f}")


if __name__ == "__main__":
    main()
