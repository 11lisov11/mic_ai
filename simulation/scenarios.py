"""
Scenario helpers: return (omega_ref(t), load_torque(t)).
"""

from __future__ import annotations

import math
from typing import Callable

from config.env import EnvConfig


Scenario = tuple[Callable[[float], float], Callable[[float], float]]


def get_scenario(name: str, env: EnvConfig) -> Scenario:
    name_raw = str(name)
    base_name = name_raw
    omega_pu = None
    if ':' in name_raw:
        base_name, pu_text = name_raw.split(':', 1)
        try:
            omega_pu = float(pu_text)
        except ValueError:
            omega_pu = None
    base_name = base_name.strip()
    omega_nom = 2.0 * math.pi * env.scalar_vf.f_max / env.motor.p
    load_const = env.sim.load_torque

    if base_name == 'speed_step':
        t_step = 0.1 * env.sim.t_end
        target_pu = 0.8 if omega_pu is None else omega_pu
        omega_target = target_pu * omega_nom

        def omega_ref(t: float) -> float:
            return 0.0 if t < t_step else omega_target

        def load_torque(t: float) -> float:
            return load_const

    elif base_name == 'ramp':
        t_ramp = max(env.sim.t_end * 0.6, env.sim.dt)
        target_pu = 1.0 if omega_pu is None else omega_pu
        omega_target = target_pu * omega_nom

        def omega_ref(t: float) -> float:
            if t >= t_ramp:
                return omega_target
            return omega_target * (t / t_ramp)

        def load_torque(t: float) -> float:
            return load_const

    elif base_name == 'load_step':
        t_step = 0.3 * env.sim.t_end
        target_pu = 0.8 if omega_pu is None else omega_pu
        omega_target = target_pu * omega_nom

        def omega_ref(t: float) -> float:
            return omega_target

        def load_torque(t: float) -> float:
            return 0.0 if t < t_step else load_const

    elif base_name == 'start_stop':
        t_up = max(env.sim.t_end * 0.2, env.sim.dt)
        t_down = max(env.sim.t_end * 0.2, env.sim.dt)
        t_hold = max(env.sim.t_end - t_up - t_down, env.sim.dt)
        t_down_start = t_up + t_hold
        target_pu = 1.0 if omega_pu is None else omega_pu
        omega_target = target_pu * omega_nom

        def omega_ref(t: float) -> float:
            if t < t_up:
                return omega_target * (t / t_up)
            if t < t_down_start:
                return omega_target
            if t < t_down_start + t_down:
                return omega_target * (1.0 - (t - t_down_start) / t_down)
            return 0.0

        def load_torque(t: float) -> float:
            return load_const

    else:
        raise ValueError(f"Unknown scenario '{name}'")

    return omega_ref, load_torque


__all__ = ['get_scenario']
