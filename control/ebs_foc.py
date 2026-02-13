"""
Energy-Budget Supervisory Control (EBS) on top of classic FOC.
The supervisor adapts id_ref to meet an efficiency target under load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from control.vector_foc import PI
from models.transformations import abc_to_dq


@dataclass
class EbsParams:
    eta_target: float = 0.8           # target efficiency (mechanical / electrical)
    window_sec: float = 0.2           # energy window
    id_ref_min: float | None = None
    id_ref_max: float | None = None
    id_ref_step: float | None = None
    err_tol_rel: float = 0.05         # allowed speed error ratio
    err_tol_abs: float = 0.0          # optional absolute error threshold
    min_omega_pu: float = 0.1         # below this, use high flux
    budget_margin: float = 0.03       # deadband on budget ratio
    max_di_dt: float = 500.0          # A/s slew-limit


class EbsFocController:
    """
    FOC with an energy-budget supervisor.
    The supervisor adapts id_ref to keep E_in <= E_mech / eta_target
    while respecting speed error bounds.
    """

    def __init__(
        self,
        params: FocParams,
        motor_params: MotorParams,
        dt: float,
        omega_base: float,
        ebs_params: EbsParams | None = None,
    ) -> None:
        self.params = params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.dt = float(dt)
        self.omega_base = float(max(omega_base, 1e-6))

        self.ebs = ebs_params or EbsParams()
        self._init_ebs_defaults()

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

        # energy window accumulators
        self._t_acc = 0.0
        self._e_in = 0.0
        self._e_mech = 0.0
        self._id_ref = float(self.id_ref_max)

    def _init_ebs_defaults(self) -> None:
        id_max = float(self.ebs.id_ref_max) if self.ebs.id_ref_max is not None else float(self.params.id_ref or 0.0)
        if id_max <= 0.0:
            id_max = float(NAMEPLATE_ID_REF)
        id_min = float(self.ebs.id_ref_min) if self.ebs.id_ref_min is not None else 0.4 * id_max
        self.id_ref_max = id_max
        self.id_ref_min = max(0.0, min(id_min, id_max))
        step = float(self.ebs.id_ref_step) if self.ebs.id_ref_step is not None else 0.02 * id_max
        self.id_ref_step = max(step, 1e-4)

        self.window_sec = float(max(self.ebs.window_sec, self.dt))
        self.eta_target = float(max(min(self.ebs.eta_target, 0.98), 0.1))
        self.err_tol_rel = float(max(self.ebs.err_tol_rel, 0.0))
        self.err_tol_abs = float(max(self.ebs.err_tol_abs, 0.0))
        self.min_omega_pu = float(max(self.ebs.min_omega_pu, 0.0))
        self.budget_margin = float(max(self.ebs.budget_margin, 0.0))
        self.max_di_dt = float(max(self.ebs.max_di_dt, 0.0))

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self._t_acc = 0.0
        self._e_in = 0.0
        self._e_mech = 0.0
        self._id_ref = float(self.id_ref_max)

    def update_energy(self, p_in: float, p_mech: float, omega_ref: float, omega_m: float) -> None:
        # freeze and reset when speed is too low
        if abs(omega_ref) < self.min_omega_pu * self.omega_base:
            self._id_ref = float(self.id_ref_max)
            self._t_acc = 0.0
            self._e_in = 0.0
            self._e_mech = 0.0
            return

        err = abs(omega_ref - omega_m)
        err_limit = max(self.err_tol_abs, self.err_tol_rel * max(abs(omega_ref), 1e-6))
        if err > err_limit:
            # prioritize accuracy -> add flux
            self._id_ref = min(self.id_ref_max, self._id_ref + self.id_ref_step)
            self._t_acc = 0.0
            self._e_in = 0.0
            self._e_mech = 0.0
            return

        self._e_in += max(p_in, 0.0) * self.dt
        self._e_mech += max(p_mech, 0.0) * self.dt
        self._t_acc += self.dt

        if self._t_acc < self.window_sec:
            return

        # compare energy to budget
        if self._e_mech <= 1e-9:
            # no useful mechanical output -> avoid starving flux
            self._id_ref = float(self.id_ref_max)
        else:
            e_budget = self._e_mech / self.eta_target
            ratio = self._e_in / max(e_budget, 1e-9)
            if ratio > 1.0 + self.budget_margin:
                self._id_ref = max(self.id_ref_min, self._id_ref - self.id_ref_step)
            elif ratio < 1.0 - self.budget_margin:
                # we are below budget: allow slightly lower flux to search for savings
                self._id_ref = max(self.id_ref_min, self._id_ref - 0.5 * self.id_ref_step)
            else:
                # within budget band -> keep
                self._id_ref = min(self.id_ref_max, max(self.id_ref_min, self._id_ref))

        self._t_acc = 0.0
        self._e_in = 0.0
        self._e_mech = 0.0

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
            "id_ref_ebs": float(self._id_ref),
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["EbsFocController", "EbsParams"]
