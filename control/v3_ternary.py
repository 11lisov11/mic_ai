"""
V3 (TDN V3) controller: ternary decision logic for speed loop + PI current loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from models.transformations import abc_to_dq
from control.vector_foc import PI


@dataclass
class V3Params:
    # Speed decision logic
    deadband: float | None = None        # absolute deadband in rad/s
    deadband_rel: float = 0.01           # if deadband is None, use deadband_rel * omega_base
    leak_rate: float = 1.0               # 1/s for ternary integrator leakage
    iu_limit: float = 0.6                # limit for ternary integrator state
    kp_u: float | None = None            # gain for ternary decision u (A)
    ki_u: float | None = None            # gain for ternary integrator (A / unit Iu)

    # Torque/flux shaping
    iq_limit: float | None = None        # absolute iq limit (A)
    iq_low: float | None = None          # threshold for low flux (A)
    iq_high: float | None = None         # threshold for high flux (A)
    id_ref_low: float | None = None      # low flux id reference (A)
    id_ref_mid: float | None = None      # mid flux id reference (A)
    id_ref_high: float | None = None     # high flux id reference (A)

    max_di_dt: float = 500.0             # A/s slew-rate for id/iq refs


class V3Controller:
    """
    V3 controller: ternary speed decision -> ternary integrator -> iq_ref
    + ternary flux selection -> id_ref, with inner PI current loops.
    """

    def __init__(
        self,
        foc_params: FocParams,
        motor_params: MotorParams,
        dt: float,
        omega_base: float,
        v3_params: V3Params | None = None,
    ) -> None:
        self.params = foc_params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.I_n = float(getattr(motor_params, "I_n", 0.0) or 0.0)
        self.dt = float(dt)
        self.omega_base = float(max(omega_base, 1e-6))

        self.v3 = v3_params or V3Params()
        self._derive_v3_defaults()

        self.pi_id = PI(foc_params.kp_id, foc_params.ki_id, dt, limit=foc_params.v_limit)
        self.pi_iq = PI(foc_params.kp_iq, foc_params.ki_iq, dt, limit=foc_params.v_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

        self.u = 0.0
        self.iu = 0.0

    def _derive_v3_defaults(self) -> None:
        iq_limit = self.v3.iq_limit
        if iq_limit is None or iq_limit <= 0.0:
            iq_limit = float(self.params.iq_limit or 0.0)
        if iq_limit <= 0.0:
            iq_limit = float(getattr(self.params, "iq_limit", 0.0) or 0.0)
        if iq_limit <= 0.0:
            iq_limit = float(self.I_n if self.I_n > 0.0 else 1.0)
        self.iq_limit = float(iq_limit)

        self.deadband = (
            float(self.v3.deadband)
            if self.v3.deadband is not None
            else float(self.v3.deadband_rel) * self.omega_base
        )
        self.leak_rate = float(max(self.v3.leak_rate, 0.0))
        self.iu_limit = float(max(self.v3.iu_limit, 0.0))

        self.kp_u = float(self.v3.kp_u) if self.v3.kp_u is not None else 0.6 * self.iq_limit
        self.ki_u = float(self.v3.ki_u) if self.v3.ki_u is not None else 2.5 * self.iq_limit

        self.iq_low = float(self.v3.iq_low) if self.v3.iq_low is not None else 0.3 * self.iq_limit
        self.iq_high = float(self.v3.iq_high) if self.v3.iq_high is not None else 0.7 * self.iq_limit

        id_high = float(self.v3.id_ref_high) if self.v3.id_ref_high is not None else float(self.params.id_ref or 0.0)
        if id_high <= 0.0:
            id_high = float(NAMEPLATE_ID_REF)
        self.id_ref_high = id_high
        self.id_ref_mid = float(self.v3.id_ref_mid) if self.v3.id_ref_mid is not None else 0.75 * id_high
        self.id_ref_low = float(self.v3.id_ref_low) if self.v3.id_ref_low is not None else 0.5 * id_high

        self.max_di_dt = float(max(self.v3.max_di_dt, 0.0))

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self.u = 0.0
        self.iu = 0.0

    def _ternary(self, error: float) -> float:
        if error > self.deadband:
            return 1.0
        if error < -self.deadband:
            return -1.0
        return 0.0

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

        e_speed = float(omega_ref - omega_m)
        self.u = self._ternary(e_speed)
        leak = max(0.0, 1.0 - self.leak_rate * self.dt)
        self.iu = leak * self.iu + self.u * self.dt
        if self.iu_limit > 0.0:
            self.iu = max(-self.iu_limit, min(self.iu_limit, self.iu))

        i_q_ref = self.kp_u * self.u + self.ki_u * self.iu
        i_q_ref = max(-self.iq_limit, min(self.iq_limit, i_q_ref))

        abs_iq = abs(i_q_ref)
        if abs_iq > self.iq_high:
            i_d_ref = self.id_ref_high
        elif abs_iq > self.iq_low:
            i_d_ref = self.id_ref_mid
        else:
            i_d_ref = self.id_ref_low

        if self.max_di_dt > 0.0:
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
            "u": float(self.u),
            "iu": float(self.iu),
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["V3Controller", "V3Params"]
