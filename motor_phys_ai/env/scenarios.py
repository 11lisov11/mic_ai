# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Callable

from config.env import EnvConfig

Scenario = tuple[Callable[[float], float], Callable[[float], float]]


def get_scenario(name: str, env: EnvConfig) -> Scenario:
    name_raw = str(name)
    base_name = name_raw
    omega_pu = None
    if ":" in name_raw:
        base_name, pu_text = name_raw.split(":", 1)
        try:
            omega_pu = float(pu_text)
        except ValueError:
            omega_pu = None
    base_name = base_name.strip()

    omega_nom = 2.0 * math.pi * env.scalar_vf.f_max / env.motor.p
    load_const = float(env.sim.load_torque)
    target_pu = 0.8 if omega_pu is None else omega_pu
    omega_target = target_pu * omega_nom

    if base_name == "step":
        t_step = 0.1 * env.sim.t_end

        def omega_ref(t: float) -> float:
            return 0.0 if t < t_step else omega_target

        def load_torque(t: float) -> float:
            return load_const

    elif base_name == "load":
        t_step = 0.3 * env.sim.t_end

        def omega_ref(t: float) -> float:
            return omega_target

        def load_torque(t: float) -> float:
            return 0.0 if t < t_step else load_const

    elif base_name == "drift":
        t_start = 0.2 * env.sim.t_end
        t_end = env.sim.t_end
        load_end = 1.5 * load_const

        def omega_ref(t: float) -> float:
            return omega_target

        def load_torque(t: float) -> float:
            if t <= t_start:
                return load_const
            if t >= t_end:
                return load_end
            alpha = (t - t_start) / max(t_end - t_start, env.sim.dt)
            return load_const + alpha * (load_end - load_const)

    else:
        raise ValueError(f"Unknown scenario '{name}'")

    return omega_ref, load_torque


__all__ = ["get_scenario"]
