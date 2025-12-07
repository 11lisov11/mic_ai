from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


def _episode_metrics(env: InductionMotorEnv, max_steps: int) -> Tuple[float, float]:
    obs = env.reset()
    done = False
    cum_err = 0.0
    cum_current = 0.0
    steps = 0
    while not done and steps < max_steps:
        obs, _, done, info = env.step(None)
        omega_meas = float(obs[0]) if hasattr(obs, "__len__") and len(obs) > 0 else float(info.get("omega_meas", 0.0))
        omega_ref = float(info.get("omega_ref", 0.0))
        err = abs(omega_ref - omega_meas)
        i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
        i_rms = float(np.sqrt(np.mean(np.square(i_abc))))
        if not np.isfinite(err) or not np.isfinite(i_rms):
            break
        cum_err += err
        cum_current += i_rms
        steps += 1
    steps = max(steps, 1)
    return cum_err / steps, cum_current / steps


def compute_baseline(env_config_path: str, episodes: int) -> Dict[str, float]:
    env = make_env_from_config(env_config_path)
    env_cfg = env.env_config
    errors: List[float] = []
    currents: List[float] = []
    omega_ref_target = 2.0 * np.pi * 15.0 / max(getattr(env_cfg.motor, "p", 1), 1)
    max_steps = int(max(0.2 / env_cfg.sim.dt, 1))
    for _ in range(episodes):
        base_env = InductionMotorEnv(env_cfg)
        base_env.omega_ref_func = lambda _t, ref=omega_ref_target: ref
        base_env.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)
        j_speed, j_current = _episode_metrics(base_env, max_steps)
        if np.isfinite(j_speed) and np.isfinite(j_current):
            errors.append(j_speed)
            currents.append(j_current)
    if not errors or not currents:
        raise RuntimeError(f"Baseline simulation failed for {env_config_path}")
    baseline_speed = float(np.mean(errors))
    baseline_current = float(np.mean(currents))
    ext_scale = float(max(baseline_speed + baseline_current, 1.0))
    return {
        "baseline_speed_err": baseline_speed,
        "baseline_current_rms": baseline_current,
        "ext_scale": ext_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline FOC metrics for reference comparison.")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["config/env_demo_true_motor1.py", "config/env_demo_true_motor2.py"],
        help="List of env config paths",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per env")
    parser.add_argument("--output", type=str, default="outputs/baseline_foc.json", help="Where to store JSON results")
    args = parser.parse_args()

    results: Dict[str, Dict[str, float]] = {}
    for env_path in args.envs:
        cfg_path = Path(env_path).as_posix()
        metrics = compute_baseline(cfg_path, args.episodes)
        name = Path(env_path).stem
        results[name] = metrics
        print(f"{name}: J_speed={metrics['baseline_speed_err']:.5f}, J_current={metrics['baseline_current_rms']:.5f}, ext_scale={metrics['ext_scale']:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
