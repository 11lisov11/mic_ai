"""
Loss-Model Control (LMC) on top of classic FOC.
Compute id_ref that minimizes estimated loss for a given torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from control.vector_foc import PI
from models.transformations import abc_to_dq


@dataclass
class LmcParams:
    id_ref_min: float | None = None
    id_ref_max: float | None = None
    max_di_dt: float = 500.0
    k_fe: float = 1e-4


class LmcFocController:
    """
    FOC + analytical loss-minimizing id_ref.
    Uses torque estimate to compute id_opt that minimizes:
        P_loss ≈ Rs*(id^2 + iq^2) + k_fe*omega_e^2*(Lm*id)^2
    with torque constraint T ≈ k_t * id * iq.
    """

    def __init__(
        self,
        params: FocParams,
        motor_params: MotorParams,
        dt: float,
        omega_base: float,
        lmc_params: LmcParams | None = None,
    ) -> None:
        self.params = params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Rs = float(getattr(motor_params, "Rs", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.Lm = float(getattr(motor_params, "Lm", 0.0))
        self.dt = float(dt)
        self.omega_base = float(max(omega_base, 1e-6))

        self.lmc = lmc_params or LmcParams()
        self._init_lmc_defaults()

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

    def _init_lmc_defaults(self) -> None:
        id_max = float(self.lmc.id_ref_max) if self.lmc.id_ref_max is not None else float(self.params.id_ref or 0.0)
        if id_max <= 0.0:
            id_max = float(NAMEPLATE_ID_REF)
        id_min = float(self.lmc.id_ref_min) if self.lmc.id_ref_min is not None else 0.4 * id_max
        self.id_ref_max = id_max
        self.id_ref_min = max(0.0, min(id_min, id_max))
        self.max_di_dt = float(max(self.lmc.max_di_dt, 0.0))
        self.k_fe = float(max(self.lmc.k_fe, 0.0))

        # torque constant approximation for IM in rotor-flux orientation
        # T ≈ k_t * id * iq
        if self.Lr > 1e-9:
            self.k_t = 1.5 * self.p * (self.Lm * self.Lm / self.Lr)
        else:
            self.k_t = 1.0

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

    def _id_opt_from_torque(self, torque: float, omega_e: float) -> float:
        # loss model: a*id^2 + b/id^2 -> id_opt = (b/a)^(1/4)
        a = max(self.Rs + self.k_fe * (omega_e * omega_e) * (self.Lm * self.Lm), 1e-9)
        b = self.Rs * (torque / max(self.k_t, 1e-9)) ** 2
        id_opt = (b / a) ** 0.25 if b > 0.0 else self.id_ref_min
        return max(self.id_ref_min, min(self.id_ref_max, id_opt))

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

        # use measured torque_e to compute optimal id_ref
        id_ref = self._id_opt_from_torque(torque_e, omega_m)

        # Slew limit
        max_delta = self.max_di_dt * self.dt
        i_q_ref = max(self.last_iq_ref - max_delta, min(self.last_iq_ref + max_delta, i_q_ref))
        id_ref = max(self.last_id_ref - max_delta, min(self.last_id_ref + max_delta, id_ref))
        self.last_iq_ref = float(i_q_ref)
        self.last_id_ref = float(id_ref)

        e_id = id_ref - i_d
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
            omega_slip = (self.Rr / self.Lr) * (i_q_ref / max(abs(id_ref), eps))
        else:
            omega_slip = 0.0
        omega_syn = self.p * omega_m + omega_slip
        self.theta_e = theta_e + omega_syn * self.dt
        self.omega_syn = omega_syn

        info = {
            "i_d_ref": float(id_ref),
            "i_q_ref": float(i_q_ref),
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["LmcFocController", "LmcParams"]
