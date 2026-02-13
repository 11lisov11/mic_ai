"""
Hybrid controller: switch between FOC and V3 based on load and speed error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from config.env import FocParams, MotorParams
from control.vector_foc import FocController
from control.v3_ternary import V3Controller, V3Params


@dataclass
class HybridParams:
    err_tol_rel: float = 0.02
    err_tol_abs: float = 0.0
    min_omega_pu: float = 0.1
    load_low_ratio: float = 0.6   # use V3 when |load| <= ratio * load_nom


class HybridV3FocController:
    """
    Use V3 in low-load steady regimes, fallback to FOC when error is large.
    """

    def __init__(
        self,
        foc_params: FocParams,
        motor_params: MotorParams,
        dt: float,
        omega_base: float,
        load_nom: float,
        v3_params: V3Params | None = None,
        hybrid_params: HybridParams | None = None,
    ) -> None:
        self.foc = FocController(foc_params, motor_params, dt)
        self.v3 = V3Controller(foc_params, motor_params, dt, omega_base, v3_params)
        self.omega_base = float(max(omega_base, 1e-6))
        self.load_nom = float(max(abs(load_nom), 1e-6))
        self.hy = hybrid_params or HybridParams()
        self.last_mode = "FOC"
        self.theta_e = 0.0
        self.omega_syn = 0.0

    def reset(self) -> None:
        self.foc.reset()
        self.v3.reset()
        self.last_mode = "FOC"
        self.theta_e = 0.0
        self.omega_syn = 0.0

    def _select_mode(self, omega_ref: float, omega_m: float, load_torque: float) -> str:
        err = abs(omega_ref - omega_m)
        err_limit = max(self.hy.err_tol_abs, self.hy.err_tol_rel * max(abs(omega_ref), 1e-6))
        low_speed_ok = abs(omega_ref) >= self.hy.min_omega_pu * self.omega_base
        low_load = abs(load_torque) <= self.hy.load_low_ratio * self.load_nom
        if low_speed_ok and low_load and err <= err_limit:
            return "V3"
        return "FOC"

    def step(
        self,
        t: float,
        omega_ref: float,
        omega_m: float,
        i_abc: Tuple[float, float, float],
        torque_e: float,
        theta_mech: float,
        load_torque: float,
    ) -> Tuple[float, float, float, float, dict]:
        # update both controllers to keep states consistent
        v_d_f, v_q_f, theta_f, omega_syn_f, info_f = self.foc.step(
            t=t,
            omega_ref=omega_ref,
            omega_m=omega_m,
            i_abc=i_abc,
            torque_e=torque_e,
            theta_mech=theta_mech,
        )
        v_d_v, v_q_v, theta_v, omega_syn_v, info_v = self.v3.step(
            t=t,
            omega_ref=omega_ref,
            omega_m=omega_m,
            i_abc=i_abc,
            torque_e=torque_e,
            theta_mech=theta_mech,
        )

        mode = self._select_mode(omega_ref, omega_m, load_torque)
        self.last_mode = mode
        if mode == "V3":
            self.theta_e = theta_v
            self.omega_syn = omega_syn_v
            info = dict(info_v)
            info["hybrid_mode"] = "V3"
            return v_d_v, v_q_v, theta_v, omega_syn_v, info
        self.theta_e = theta_f
        self.omega_syn = omega_syn_f
        info = dict(info_f)
        info["hybrid_mode"] = "FOC"
        return v_d_f, v_q_f, theta_f, omega_syn_f, info


__all__ = ["HybridV3FocController", "HybridParams"]
