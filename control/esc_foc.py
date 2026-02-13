"""
FOC with extremum-seeking adaptation of id_ref to reduce input power under load.
New for this project: slow energy-seeking loop on top of classic FOC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from control.vector_foc import PI
from models.transformations import abc_to_dq


@dataclass
class EscParams:
    id_ref_min: float | None = None
    id_ref_max: float | None = None
    id_ref_step: float | None = None
    adapt_interval: float = 0.05      # s
    err_tol_rel: float = 0.02         # allow speed error up to 2% before forcing high flux
    err_tol_abs: float = 0.0          # optional absolute error threshold (rad/s)
    min_omega_pu: float = 0.1         # below this, use high flux


class EscFocController:
    """
    Classic FOC (PI speed + PI currents) with a slow hill-climb on id_ref
    to reduce positive electrical power when speed is steady.
    """

    def __init__(
        self,
        params: FocParams,
        motor_params: MotorParams,
        dt: float,
        omega_base: float,
        esc_params: EscParams | None = None,
    ) -> None:
        self.params = params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.dt = float(dt)
        self.omega_base = float(max(omega_base, 1e-6))

        self.esc = esc_params or EscParams()
        self._init_esc_defaults()

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self.max_di_dt = 500.0  # A/s

        # ESC internal state
        self._id_ref = float(self.id_ref_max)
        self._dir = -1.0
        self._timer = 0.0
        self._p_acc = 0.0
        self._p_count = 0
        self._p_prev: float | None = None

    def _init_esc_defaults(self) -> None:
        id_max = float(self.esc.id_ref_max) if self.esc.id_ref_max is not None else float(self.params.id_ref or 0.0)
        if id_max <= 0.0:
            id_max = float(NAMEPLATE_ID_REF)
        id_min = float(self.esc.id_ref_min) if self.esc.id_ref_min is not None else 0.6 * id_max

        self.id_ref_max = id_max
        self.id_ref_min = max(0.0, min(id_min, id_max))
        step = float(self.esc.id_ref_step) if self.esc.id_ref_step is not None else 0.02 * id_max
        self.id_ref_step = max(step, 1e-4)

        self.adapt_interval = float(max(self.esc.adapt_interval, self.dt))
        self.err_tol_rel = float(max(self.esc.err_tol_rel, 0.0))
        self.err_tol_abs = float(max(self.esc.err_tol_abs, 0.0))
        self.min_omega_pu = float(max(self.esc.min_omega_pu, 0.0))

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

        self._id_ref = float(self.id_ref_max)
        self._dir = -1.0
        self._timer = 0.0
        self._p_acc = 0.0
        self._p_count = 0
        self._p_prev = None

    def _update_esc(self, omega_ref: float, omega_m: float, p_in: float) -> None:
        # Freeze adaptation when speed error is large or speed is too low.
        err = abs(omega_ref - omega_m)
        err_limit = max(self.err_tol_abs, self.err_tol_rel * max(abs(omega_ref), 1e-6))
        if abs(omega_ref) < self.min_omega_pu * self.omega_base or err > err_limit:
            self._id_ref = float(self.id_ref_max)
            self._timer = 0.0
            self._p_acc = 0.0
            self._p_count = 0
            self._p_prev = None
            return

        # Accumulate positive power estimate over the interval.
        self._p_acc += max(p_in, 0.0)
        self._p_count += 1
        self._timer += self.dt
        if self._timer < self.adapt_interval:
            return

        p_mean = self._p_acc / max(self._p_count, 1)
        if self._p_prev is not None:
            # If power got worse, flip search direction.
            if p_mean > self._p_prev:
                self._dir *= -1.0
        self._p_prev = p_mean

        # Update id_ref with small step.
        self._id_ref += self._dir * self.id_ref_step
        self._id_ref = max(self.id_ref_min, min(self.id_ref_max, self._id_ref))

        # Reset window
        self._timer = 0.0
        self._p_acc = 0.0
        self._p_count = 0

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

        # Estimate instantaneous input power in dq
        # v_d/v_q are not yet known; approximate with v ~ PI outputs (use previous loop)
        # Use i_d/i_q and last PI outputs (close enough for adaptation).
        p_in_est = 1.5 * (self.last_id_ref * i_d + self.last_iq_ref * i_q)
        self._update_esc(omega_ref, omega_m, p_in_est)

        i_d_ref = self._id_ref

        # Slew limit
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
            "id_ref_esc": float(self._id_ref),
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["EscFocController", "EscParams"]
