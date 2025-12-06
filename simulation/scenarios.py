"""
Built-in simulation scenarios for reference commands and load torque.
"""

from __future__ import annotations

import math
from typing import Callable

from config.env import EnvConfig


Scenario = tuple[Callable[[float], float], Callable[[float], float]]


def get_scenario(name: str, env: EnvConfig) -> Scenario:
    """
    Return callable pair (omega_ref(t), load_torque(t)) for the given scenario name.
    """
    omega_nom = 2.0 * math.pi * env.scalar_vf.f_max / env.motor.p
    load_const = env.sim.load_torque

    if name == "speed_step":
        t_step = 0.1 * env.sim.t_end
        omega_target = 0.8 * omega_nom

        def omega_ref(t: float) -> float:
            return 0.0 if t < t_step else omega_target

        def load_torque(t: float) -> float:
            return load_const

    elif name == "ramp":
        t_ramp = max(env.sim.t_end * 0.6, env.sim.dt)

        def omega_ref(t: float) -> float:
            if t >= t_ramp:
                return omega_nom
            return omega_nom * (t / t_ramp)

        def load_torque(t: float) -> float:
            return load_const

    else:
        raise ValueError(f"Unknown scenario '{name}'")

    return omega_ref, load_torque


__all__ = ["get_scenario"]

