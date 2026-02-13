"""
FOC with load-to-flux map (LMAP): choose id_ref as a function of load torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from config.env import FocParams, MotorParams, NAMEPLATE_ID_REF
from control.vector_foc import PI
from models.transformations import abc_to_dq


@dataclass
class LoadMapParams:
    load_points: Tuple[float, ...]
    id_ref_points: Tuple[float, ...]
    max_di_dt: float = 500.0


class LoadMapFocController:
    """
    Classic FOC with id_ref taken from a load->flux map.
    """

    def __init__(
        self,
        params: FocParams,
        motor_params: MotorParams,
        dt: float,
        load_map: LoadMapParams | None = None,
    ) -> None:
        self.params = params
        self.p = motor_params.p
        self.Rr = float(getattr(motor_params, "Rr", 0.0))
        self.Lr = float(getattr(motor_params, "Lr_sigma", 0.0) + getattr(motor_params, "Lm", 0.0))
        self.dt = float(dt)

        self.load_map = load_map or self._default_map(params)

        self.pi_id = PI(params.kp_id, params.ki_id, dt, limit=params.v_limit)
        self.pi_iq = PI(params.kp_iq, params.ki_iq, dt, limit=params.v_limit)
        self.pi_speed = PI(params.kp_speed, params.ki_speed, dt, limit=params.iq_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0
        self.max_di_dt = float(self.load_map.max_di_dt)

    def _default_map(self, params: FocParams) -> LoadMapParams:
        id_high = float(params.id_ref or 0.0)
        if id_high <= 0.0:
            id_high = float(NAMEPLATE_ID_REF)
        id_low = 0.5 * id_high
        return LoadMapParams(
            load_points=(0.0, 1.0),
            id_ref_points=(id_low, id_high),
        )

    def set_map(self, load_points: Iterable[float], id_ref_points: Iterable[float]) -> None:
        loads = tuple(float(x) for x in load_points)
        ids = tuple(float(x) for x in id_ref_points)
        if len(loads) != len(ids) or len(loads) < 2:
            raise ValueError("load_map requires at least 2 points with equal length.")
        self.load_map = LoadMapParams(load_points=loads, id_ref_points=ids, max_di_dt=self.max_di_dt)

    def reset(self) -> None:
        self.pi_id.reset()
        self.pi_iq.reset()
        self.pi_speed.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_iq_ref = 0.0
        self.last_id_ref = 0.0

    def _interp_id_ref(self, load_torque: float) -> float:
        loads = self.load_map.load_points
        ids = self.load_map.id_ref_points
        x = abs(float(load_torque))
        if x <= loads[0]:
            return ids[0]
        if x >= loads[-1]:
            return ids[-1]
        for i in range(len(loads) - 1):
            if loads[i] <= x <= loads[i + 1]:
                t = (x - loads[i]) / max(loads[i + 1] - loads[i], 1e-9)
                return ids[i] + t * (ids[i + 1] - ids[i])
        return ids[-1]

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

        # Use torque_e as proxy for load torque (steady-state approx).
        i_d_ref = self._interp_id_ref(torque_e)

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
        }
        return v_d, v_q, theta_e, omega_syn, info


__all__ = ["LoadMapFocController", "LoadMapParams"]
