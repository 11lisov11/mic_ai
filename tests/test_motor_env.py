# -*- coding: utf-8 -*-
from dataclasses import replace

import numpy as np

from config.env_demo_true_motor1_nominal import ENV
from motor_phys_ai.controllers.pi import PIController
from motor_phys_ai.env.motor_env import MotorEnv


def test_motor_env_run_outputs() -> None:
    sim_cfg = replace(ENV.sim, dt=0.002, t_end=0.02, scenario_name="speed_step")
    env_cfg = replace(ENV, sim=sim_cfg)
    env = MotorEnv(env_cfg)

    ctrl = PIController(kp=0.5, ki=2.0, dt=env.dt, iq_limit=1.5)
    series = env.run(ctrl, scenario_name="step")

    assert series["t"].size > 0
    for key in ("omega", "omega_ref", "i_q", "p_el", "i_rms"):
        assert key in series
        assert series[key].size == series["t"].size

    assert np.any(np.abs(series["p_el"]) > 0.0)
