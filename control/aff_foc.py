"""
Adaptive Flux FOC (AFF): classic FOC with continuous flux scheduling
based on torque-producing current magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from control.vector_foc import PI
from models.transformations import abc_to_dq


@dataclass
class AffParams:
    id_ref_min: float | None = None
    id_ref_max: float | None = None
    iq_limit: float | None = None
    max_di_dt: float = 500.0


class AffFocController:
    """
    FOC with continuous flux scheduling:
    id_ref = id_min + k * |iq_ref| (clamped to [id_min, id_max]).
    """

    def __init__(
        self,
        params: FocParams,
        motor_params: MotorParams,
        dt: float,
        aff_params: AffParams | None = None,
    ) -> None:
        self.params = params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.dt = float(dt)

        self.aff = aff_params or AffParams()
        self._init_aff_defaults()

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self.max_di_dt = float(self.aff.max_di_dt)

    def _init_aff_defaults(self) -> None:
        id_max = float(self.aff.id_ref_max) if self.aff.id_ref_max is not None else float(self.params.id_ref or 0.0)
        if id_max <= 0.0:
            id_max = float(NAMEPLATE_ID_REF)
        id_min = float(self.aff.id_ref_min) if self.aff.id_ref_min is not None else 0.5 * id_max
        self.id_ref_max = id_max
        self.id_ref_min = max(0.0, min(id_min, id_max))

        iq_limit = float(self.aff.iq_limit) if self.aff.iq_limit is not None else float(self.params.iq_limit or 0.0)
        if iq_limit <= 0.0:
            iq_limit = 1.0
        self.iq_limit = iq_limit

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
        theta_mech: float,
    ) -> Tuple[float, float, float, float, dict]:
        i_a, i_b, i_c = i_abc
        theta_e = self.theta_e
        i_d, i_q = abc_to_dq(i_a, i_b, i_c, theta_e)

        e_speed = omega_ref - omega_m
        i_q_ref = self.pi_speed.step(e_speed)
        if self.params.iq_limit is not None:
            i_q_ref = max(-self.params.iq_limit, min(self.params.iq_limit, i_q_ref))

        # Continuous flux scheduling
        k = (self.id_ref_max - self.id_ref_min) / max(self.iq_limit, 1e-6)
        i_d_ref = self.id_ref_min + k * abs(i_q_ref)
        i_d_ref = max(self.id_ref_min, min(self.id_ref_max, i_d_ref))

        # Slew limiting
        max_delta = self.max_di_dt * self.dt
        i_q_ref = max(self.last_iq_ref - max_delta, min(self.last_iq_ref + max_delta, i_q_ref))
        i_d_ref = max(self.last_id_ref - max_delta, min(self.last_id_ref + max_delta, i_d_ref))
        self.last_iq_ref = float(i_q_ref)
        self.last_id_ref = float(i_d_ref)

        e_id = i_d_ref - i_d
        e_iq = i_q_ref - i_q
        v_d = self.pi_id.step(e_id)
        v_q = self.pi_iq.step(e_iq)

        if self.params.v_limit is not None:
            mag = math.hypot(v_d, v_q)
            if mag > self.params.v_limit and mag > 0.0:
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
            "i_d_ref": float(i_d_ref),
            "i_q_ref": float(i_q_ref),
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["AffFocController", "AffParams"]
