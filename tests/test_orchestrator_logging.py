import json

import numpy as np

from bench.orchestrator import ExperimentConfig, ProfileConfig, run_experiment
from config.env import create_default_env
from sim.load_servo import ServoLoadConfig
from sim.sensors import SensorConfig


def test_orchestrator_writes_logs(tmp_path) -> None:
    env = create_default_env()
    config = ExperimentConfig(
        dt=0.01,
        duration=0.02,
        motor=env.motor,
        inverter=env.inverter,
        scalar_vf=env.scalar_vf,
        foc=env.foc,
        controller="scalar",
        speed_profile=ProfileConfig(kind="step", initial=0.0, value=10.0, step_time=0.0),
        load_profile=ProfileConfig(kind="constant", value=0.0),
        load_config=ServoLoadConfig(mode="torque", tau_load=0.05, t_max=1.0),
        sensor=SensorConfig(),
        log_root=str(tmp_path),
    )
    result = run_experiment(config, policy=None)

    assert result.npz_path is not None
    assert result.json_path is not None
    assert result.npz_path.exists()
    assert result.json_path.exists()

    with result.json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert "config" in meta
    assert "npz_path" in meta

    data = np.load(result.npz_path)
    assert data["t"].size == 2
