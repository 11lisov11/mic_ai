from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


def _omega_profile(base_val: float, step_idx: int, piecewise_steps: Sequence[int], multipliers: Sequence[float]) -> float:
    seg = 0
    for boundary in piecewise_steps:
        if step_idx >= boundary:
            seg += 1
    seg = min(seg, len(multipliers) - 1)
    return base_val * multipliers[seg]


def _run_single_episode(
    env: InductionMotorEnv,
    omega_base: float,
    episode_steps: int,
    piecewise_steps: Sequence[int],
    multipliers: Sequence[float],
) -> Dict[str, float]:
    current_ref = [omega_base * multipliers[0]]

    def _omega_ref_func(_t: float) -> float:
        return current_ref[0]

    env.omega_ref_func = _omega_ref_func
    env.load_torque_func = getattr(env, "load_torque_func", lambda _t: getattr(getattr(env, "env_config", None), "sim", None) and getattr(getattr(env, "env_config", None).sim, "load_torque", 0.0))

    obs = env.reset()
    cum_err = 0.0
    cum_i = 0.0
    omega_vals: List[float] = []
    for step in range(episode_steps):
        current_ref[0] = _omega_profile(omega_base, step, piecewise_steps, multipliers)
        obs, _, done, info = env.step(None)
        omega_meas_raw = float(obs[0]) if hasattr(obs, "__len__") and len(obs) > 0 else float(info.get("omega_meas", 0.0))
        omega_meas = float(np.nan_to_num(omega_meas_raw, nan=0.0, posinf=0.0, neginf=0.0))
        omega_ref_raw = float(info.get("omega_ref", current_ref[0]))
        omega_ref = float(np.nan_to_num(omega_ref_raw, nan=0.0, posinf=0.0, neginf=0.0))
        i_abc = np.nan_to_num(np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        i_rms = float(np.clip(np.sqrt(np.mean(np.square(i_abc))), 0.0, 1e6))
        cum_err += abs(omega_ref - omega_meas)
        cum_i += i_rms
        omega_vals.append(omega_meas)
        if done:
            break
    steps = max(len(omega_vals), 1)
    return {
        "mean_speed_error": cum_err / steps,
        "mean_current_rms": cum_i / steps,
        "steps": steps,
        "omega_mean": float(np.mean(omega_vals)) if omega_vals else 0.0,
    }


def _sanitize_result(
    res: Dict[str, float], fallback_speed: float, fallback_current: float, omega_base: float, episode_steps: int
) -> Dict[str, float]:
    ok = all(np.isfinite(val) for val in res.values()) and all(abs(val) < 1e9 for val in res.values())
    if ok:
        return res
    return {
        "mean_speed_error": float(fallback_speed),
        "mean_current_rms": float(fallback_current),
        "steps": int(res.get("steps", episode_steps)),
        "omega_mean": float(omega_base),
        "note": "fallback_values_used",
    }


def run_foc_baseline(
    config_name: str, curriculum_config: Dict[str, object], n_episodes_eval: int = 5, episode_steps: int = 400
) -> List[Dict[str, float]]:
    """
    Run FOC baseline episodes matching the ai_voltage curriculum and return episode summaries.
    """
    env_sim = make_env_from_config(str(config_name))
    env_cfg = env_sim.env_config
    omega_nominal = float(abs(getattr(env_cfg.motor, "omega_base", getattr(env_cfg.motor, "w_base", getattr(env_cfg.motor, "w_nom", 1.0)))))
    piecewise_steps = tuple(int(x) for x in curriculum_config.get("piecewise_steps", (150, 300)))
    multipliers = tuple(float(x) for x in curriculum_config.get("piecewise_multipliers", (1.0, 0.8, 1.0)))
    omega_pu_stages = list(curriculum_config.get("omega_pu_stages", [0.3, 0.5]))
    fallback_speed = float(getattr(env_cfg, "baseline_speed_err", getattr(env_cfg, "baseline_speed_error", 5.0)))
    fallback_current = float(getattr(env_cfg, "baseline_current_rms", 0.5))

    results: List[Dict[str, float]] = []
    for stage_idx, omega_pu in enumerate(omega_pu_stages):
        omega_base = omega_pu * omega_nominal
        for ep in range(n_episodes_eval):
            env = InductionMotorEnv(env_cfg)
            ep_res_raw = _run_single_episode(
                env, omega_base=omega_base, episode_steps=episode_steps, piecewise_steps=piecewise_steps, multipliers=multipliers
            )
            ep_res = _sanitize_result(
                ep_res_raw,
                fallback_speed=fallback_speed,
                fallback_current=fallback_current,
                omega_base=omega_base,
                episode_steps=episode_steps,
            )
            results.append(
                {
                    "episode": stage_idx * n_episodes_eval + ep,
                    "stage": stage_idx,
                    "omega_pu": omega_pu,
                    **ep_res,
                }
            )
    return results


def save_foc_baseline(config_name: str, curriculum_config: Dict[str, object], log_path: Path, n_episodes_eval: int = 5, episode_steps: int = 400) -> Path:
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = run_foc_baseline(config_name, curriculum_config, n_episodes_eval=n_episodes_eval, episode_steps=episode_steps)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return log_path


__all__ = ["run_foc_baseline", "save_foc_baseline"]
