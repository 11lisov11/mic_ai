"""
Entry point to run a reference simulation without RL.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure project root on path when executed as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import ENV, create_default_env
from simulation.gym_env import InductionMotorEnv


def _next_data_path(results_dir: Path) -> Path:
    """
    Generate sequential data file path: data_1.npz, data_2.npz, ...
    """
    idx = 1
    while True:
        candidate = results_dir / f"data_{idx}.npz"
        if not candidate.exists():
            return candidate
        idx += 1


def run_simulation(env_config=None) -> Path:
    if env_config is None:
        env_config = create_default_env()
    env = InductionMotorEnv(env_config)
    obs = env.reset()

    n_steps = int(env_config.sim.t_end / env_config.sim.dt)
    results = {
        "t": [],
        "omega_m": [],
        "omega_ref": [],
        "T_e": [],
        "load_torque": [],
        "i_a": [],
        "i_b": [],
        "i_c": [],
        "P_in": [],
        "P_out": [],
    }

    for _ in tqdm(range(n_steps), desc="Simulating", leave=False):
        obs, _, done, info = env.step()
        results["t"].append(env.t)
        results["omega_m"].append(float(obs[0]))
        results["omega_ref"].append(float(obs[1]))
        results["T_e"].append(float(obs[2]))
        results["i_a"].append(float(obs[3]))
        results["i_b"].append(float(obs[4]))
        results["i_c"].append(float(obs[5]))
        results["P_in"].append(float(obs[6]))
        results["P_out"].append(float(obs[7]))
        results["load_torque"].append(env_config.sim.load_torque)
        if done:
            break

    save_dir = Path("outputs") / "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = _next_data_path(save_dir)
    meta = json.dumps(asdict(env_config))
    meta_bytes = np.array(meta.encode("utf-8"), dtype=np.bytes_)
    np.savez(save_path, **{k: np.asarray(v) for k, v in results.items()}, meta=meta_bytes)
    return save_path


if __name__ == "__main__":
    path = run_simulation()
    print(f"Saved results to {path}")
