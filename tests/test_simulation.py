import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from config.env import ENV, EnvConfig, SimulationParams
from simulation.gym_env import InductionMotorEnv
from simulation.run_simulation import run_simulation


def short_env() -> EnvConfig:
    sim_override = replace(ENV.sim, t_end=0.01, dt=1e-3, save_prefix="test_run")
    return replace(ENV, sim=sim_override)


def test_gym_env_runs_and_finishes():
    env = InductionMotorEnv(short_env())
    obs = env.reset()
    assert env.observation_space.contains(obs)

    done = False
    steps = 0
    while not done and steps < 1000:
        obs, _, done, info = env.step()
        steps += 1
    assert done
    assert steps > 0
    assert env.observation_space.contains(obs)
    assert "omega_ref" in info


def test_run_simulation_saves_results(tmp_path: Path):
    env_cfg = short_env()
    env_cfg = replace(
        env_cfg,
        sim=SimulationParams(
            t_end=0.02,
            dt=1e-3,
            mode=env_cfg.sim.mode,
            scenario_name=env_cfg.sim.scenario_name,
            save_prefix="pytest_run",
            load_torque=env_cfg.sim.load_torque,
        ),
    )
    save_path = tmp_path / "outputs" / "results" / "pytest_run.npz"
    os.makedirs(save_path.parent, exist_ok=True)

    # temporarily redirect outputs
    orig_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        path = run_simulation(env_cfg)
        assert path.exists()
        data = np.load(path)
        for key in ["t", "omega_m", "omega_ref", "T_e", "i_a", "P_in"]:
            assert key in data
            assert len(data[key]) > 0
        assert "meta" in data
    finally:
        os.chdir(orig_cwd)


def test_main_cli_runs_without_plot(tmp_path: Path):
    from main import main

    orig_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result_path = main(
            [
                "--t-end",
                "0.02",
                "--dt",
                "0.001",
                "--save-prefix",
                "cli_test",
                "--mode",
                "scalar",
                "--scenario",
                "speed_step",
                "--no-plot",
            ]
        )
        assert result_path.exists()
    finally:
        os.chdir(orig_cwd)
