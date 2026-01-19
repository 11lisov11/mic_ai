# -*- coding: utf-8 -*-
from __future__ import annotations


class PhysModController:
    """
    Физически осведомленный регулятор скорости.

    i_q = u_fb + u_ff
    u_fb = Kp * e + Ki * ∫e
    u_ff = -J / Kt * omega_dot
    Kt адаптируется при |omega_dot| > threshold.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        dt: float,
        inertia_j: float,
        kt_init: float,
        iq_limit: float | None = None,
        kt_alpha: float = 0.02,
        omega_dot_threshold: float = 1.0,
        kt_min: float = 0.01,
        kt_max: float = 10.0,
        iq_min: float = 0.05,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.inertia_j = max(float(inertia_j), 1e-6)
        self.kt_hat = max(float(kt_init), 1e-6)
        self.iq_limit = None if iq_limit is None else float(iq_limit)
        self.kt_alpha = float(kt_alpha)
        self.omega_dot_threshold = float(omega_dot_threshold)
        self.kt_min = float(kt_min)
        self.kt_max = float(kt_max)
        self.iq_min = float(iq_min)
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def _clamp(self, value: float, vmin: float, vmax: float) -> float:
        return float(max(vmin, min(vmax, value)))

    def _update_kt(self, omega_dot: float, iq_cmd: float) -> None:
        if abs(omega_dot) < self.omega_dot_threshold:
            return
        if abs(iq_cmd) < self.iq_min:
            return
        kt_est = self.inertia_j * omega_dot / iq_cmd
        if kt_est <= 0.0:
            return
        kt_est = self._clamp(kt_est, self.kt_min, self.kt_max)
        self.kt_hat = (1.0 - self.kt_alpha) * self.kt_hat + self.kt_alpha * kt_est

    def step(self, omega_ref: float, omega: float, omega_dot: float) -> float:
        error = float(omega_ref - omega)
        u_fb = self.kp * error + self.integrator
        u_ff = -(self.inertia_j / max(self.kt_hat, 1e-6)) * float(omega_dot)
        u_unsat = u_fb + u_ff
        u = u_unsat

        if self.iq_limit is not None:
            u = self._clamp(u_unsat, -self.iq_limit, self.iq_limit)
            if abs(u) < self.iq_limit:
                self.integrator += error * self.ki * self.dt
        else:
            self.integrator += error * self.ki * self.dt

        self._update_kt(float(omega_dot), float(u))
        return float(u)
