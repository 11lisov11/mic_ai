# -*- coding: utf-8 -*-
from __future__ import annotations


class PIController:
    """
    Классический PI-регулятор по скорости, выход - i_q.
    """

    def __init__(self, kp: float, ki: float, dt: float, iq_limit: float | None = None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.iq_limit = None if iq_limit is None else float(iq_limit)
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def step(self, omega_ref: float, omega: float, omega_dot: float) -> float:
        error = float(omega_ref - omega)
        u_unsat = self.kp * error + self.integrator
        u = u_unsat
        if self.iq_limit is not None:
            u = max(-self.iq_limit, min(self.iq_limit, u_unsat))
            if abs(u) < self.iq_limit:
                self.integrator += error * self.ki * self.dt
        else:
            self.integrator += error * self.ki * self.dt
        return float(u)
