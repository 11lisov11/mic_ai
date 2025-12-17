"""
Векторное управление (FOC) с каскадными ПИ-контуром скорости и токов.
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
            # антинасыщение: интегрируем только если нет насыщения
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
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.dt = dt

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self.max_di_dt = 500.0  # A/s

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

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
        
        # используем синхронный угол, задаваемый ссылкой
        theta_e = self.theta_e
        i_d, i_q = abc_to_dq(i_a, i_b, i_c, theta_e)

        e_speed = omega_ref - omega_m
        i_q_ref = self.pi_speed.step(e_speed)
        if self.params.iq_limit is not None:
            i_q_ref = max(-self.params.iq_limit, min(self.params.iq_limit, i_q_ref))
        i_d_ref = self.params.id_ref if self.params.id_ref is not None else 0.5 * NAMEPLATE_I_N

        # ограничиваем скорость изменения ссылок
        max_delta = self.max_di_dt * self.dt
        i_q_ref = max(self.last_iq_ref - max_delta, min(self.last_iq_ref + max_delta, i_q_ref))
        i_d_ref = max(self.last_id_ref - max_delta, min(self.last_id_ref + max_delta, i_d_ref))
        self.last_iq_ref = i_q_ref
        self.last_id_ref = i_d_ref

        e_id = i_d_ref - i_d
        e_iq = i_q_ref - i_q

        v_d = self.pi_id.step(e_id)
        v_q = self.pi_iq.step(e_iq)

        # при необходимости ограничиваем результирующее напряжение по модулю
        if self.params.v_limit is not None:
            mag = math.hypot(v_d, v_q)
            if mag > self.params.v_limit and mag > 0:
                scale = self.params.v_limit / mag
                v_d *= scale
                v_q *= scale

        eps = 1e-6
        if self.Rr > 0.0 and self.Lr > eps:
            omega_slip = (self.Rr / self.Lr) * (i_q_ref / max(abs(i_d_ref), eps))
        else:
            omega_slip = 0.0
        omega_syn = self.p * omega_m + omega_slip
        self.theta_e = theta_e + omega_syn * self.dt
        self.omega_syn = omega_syn

        info = {
            "i_d_ref": i_d_ref,
            "i_q_ref": i_q_ref
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["FocController", "PI"]
