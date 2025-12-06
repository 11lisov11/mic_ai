"""
Open-loop scalar V/f controller with ramps, slip clamp, and voltage limits.
"""

from __future__ import annotations

import math
from typing import Tuple

from config.env import NAMEPLATE_U_LL, ScalarVfParams


class ScalarVfController:
    def __init__(self, params: ScalarVfParams, dt: float, p: int, vdc: float):
        self.params = params
        self.dt = dt
        self.p = p
        self.v_limit = vdc / math.sqrt(3.0)
        self.U_ph_nom = NAMEPLATE_U_LL / math.sqrt(3.0)

        self.theta_e = 0.0
        self.omega_e = 0.0
        self.f_cmd = 0.0
        self.v_cmd = 0.0
        self.te_filt = 0.0
        self.Te = 0.0

        # ramp rates
        self.df_max = 200.0  # Hz/s

    def reset(self) -> None:
        self.theta_e = 0.0
        self.omega_e = 0.0
        self.f_cmd = 0.0
        self.v_cmd = 0.0
        self.te_filt = 0.0

    def step(
        self,
        t: float,
        omega_ref: float,
        omega_m: float,
        i_abc: Tuple[float, float, float],
        torque_e: float,
        theta_mech: float
    ) -> Tuple[float, float, float, float, dict]:
        """
        Compute dq voltage commands from mechanical speed reference.

        Returns:
            v_d, v_q: dq voltages.
            theta_e: electrical angle.
            omega_syn: electrical angular speed.
            info: dictionary with internal state (e.g. references, filtered values).
        """
        direction = 1.0 if omega_ref >= 0 else -1.0
        f_ref = abs(omega_ref) * self.p / (2.0 * math.pi)  # Hz (electrical)

        # ramp frequency with df_max
        df = f_ref - self.f_cmd
        df = max(-self.df_max * self.dt, min(self.df_max * self.dt, df))
        self.f_cmd += df

        # enforce min/max frequency
        if self.f_cmd < self.params.f_min:
            self.f_cmd = self.params.f_min
        if self.f_cmd > self.params.f_max:
            self.f_cmd = self.params.f_max

        f_e = self.f_cmd

        # V/f law
        if f_e > 0:
            u_phase = self.params.k_vf * f_e + self.params.u_boost
        else:
            u_phase = 0.0

        # limit by nominal phase voltage
        u_phase = max(-self.U_ph_nom, min(self.U_ph_nom, u_phase))

        # electrical angle update
        self.omega_e = direction * 2.0 * math.pi * f_e
        self.theta_e += self.omega_e * self.dt

        # filtered torque (monitoring)
        alpha = 0.98
        self.Te = alpha * self.Te + (1.0 - alpha) * torque_e
        self.te_filt = self.Te

        v_d = 0.0
        v_q = direction * u_phase
        
        info = {
            "f_e": self.f_cmd,
            "te_filt": self.te_filt
        }
        return v_d, v_q, self.theta_e, self.omega_e, info


__all__ = ["ScalarVfController"]
