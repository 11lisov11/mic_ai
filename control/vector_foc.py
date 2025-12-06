"""
Vector control (FOC) with cascaded PI loops for speed and currents.
"""

from __future__ import annotations

import math
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_I_N
from models.transformations import abc_to_dq


class PI:
    def __init__(self, kp: float, ki: float, dt: float, limit: float | None = None):
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.limit = limit
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def step(self, error: float) -> float:
        u_unsat = self.kp * error + self.integrator
        if self.limit is not None:
            u = max(-self.limit, min(self.limit, u_unsat))
            # anti-windup: integrate only if not saturated
            if abs(u) < self.limit:
                self.integrator += error * self.ki * self.dt
        else:
            self.integrator += error * self.ki * self.dt
            u = u_unsat
        return u


class FocController:
    def __init__(self, params: FocParams, motor_params: MotorParams, dt: float):
        self.params = params
        self.p = motor_params.p
        self.dt = dt

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0

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
        Compute dq voltage commands using FOC.

        Returns:
            v_d, v_q: dq voltages.
            theta_e: electrical angle used for transformations.
            omega_syn: synchronous electrical speed (rad/s).
            info: debug info (current refs).
        """
        i_a, i_b, i_c = i_abc
        
        # use synchronous angle driven by reference
        theta_e = self.theta_e
        i_d, i_q = abc_to_dq(i_a, i_b, i_c, theta_e)

        e_speed = omega_ref - omega_m
        i_q_ref = self.pi_speed.step(e_speed)
        if self.params.iq_limit is not None:
            i_q_ref = max(-self.params.iq_limit, min(self.params.iq_limit, i_q_ref))
        i_d_ref = self.params.id_ref if self.params.id_ref is not None else 0.5 * NAMEPLATE_I_N

        e_id = i_d_ref - i_d
        e_iq = i_q_ref - i_q

        v_d = self.pi_id.step(e_id)
        v_q = -self.pi_iq.step(e_iq)

        # overall voltage magnitude limitation if requested
        if self.params.v_limit is not None:
            mag = math.hypot(v_d, v_q)
            if mag > self.params.v_limit and mag > 0:
                scale = self.params.v_limit / mag
                v_d *= scale
                v_q *= scale

        omega_syn = self.p * omega_ref
        self.theta_e = theta_e + omega_syn * self.dt
        self.omega_syn = omega_syn

        info = {
            "i_d_ref": i_d_ref,
            "i_q_ref": i_q_ref
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["FocController", "PI"]
