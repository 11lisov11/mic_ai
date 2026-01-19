# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from config.env import EnvConfig
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import abc_to_dq, dq_to_abc
from motor_phys_ai.env.scenarios import get_scenario


class _PI:
    def __init__(self, kp: float, ki: float, dt: float, limit: float | None = None):
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.limit = None if limit is None else float(limit)
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def step(self, error: float) -> float:
        u_unsat = self.kp * error + self.integrator
        if self.limit is None:
            self.integrator += error * self.ki * self.dt
            return float(u_unsat)
        u = max(-self.limit, min(self.limit, u_unsat))
        if abs(u) < self.limit:
            self.integrator += error * self.ki * self.dt
        return float(u)


class MotorEnv:
    """
    Среда для управления током i_q через внешний регулятор скорости.
    """

    def __init__(self, env_config: EnvConfig):
        self.env = env_config
        self.dt = float(env_config.sim.dt)
        self.t_end = float(env_config.sim.t_end)
        self.motor = InductionMotorModel(env_config.motor)
        self.inverter = IdealInverter(env_config.inverter)

        self.p = int(env_config.motor.p)
        self.Rr = float(getattr(env_config.motor, "Rr", 0.0))
        self.Lr = float(getattr(env_config.motor, "Lr_sigma", 0.0) + getattr(env_config.motor, "Lm", 0.0))
        self.v_limit = float(getattr(env_config.foc, "v_limit", 0.0) or 0.0)
        self.id_ref = float(getattr(env_config.foc, "id_ref", 0.0) or 0.0)
        self.iq_limit = float(getattr(env_config.foc, "iq_limit", 0.0) or 0.0)

        self.pi_id = _PI(env_config.foc.kp_id, env_config.foc.ki_id, self.dt, limit=self.v_limit)
        self.pi_iq = _PI(env_config.foc.kp_iq, env_config.foc.ki_iq, self.dt, limit=self.v_limit)

        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_currents_abc = (0.0, 0.0, 0.0)
        self.last_torque = 0.0
        self.t = 0.0

    def reset(self) -> None:
        self.motor = InductionMotorModel(self.env.motor)
        self.pi_id.reset()
        self.pi_iq.reset()
        self.theta_e = 0.0
        self.omega_syn = 0.0
        self.last_currents_abc = (0.0, 0.0, 0.0)
        self.last_torque = 0.0
        self.t = 0.0

    def _current_loop(self, i_q_ref: float, load_torque: float) -> Dict[str, float]:
        i_a, i_b, i_c = self.last_currents_abc
        i_d, i_q = abc_to_dq(i_a, i_b, i_c, self.theta_e)

        e_id = self.id_ref - i_d
        e_iq = i_q_ref - i_q
        v_d = self.pi_id.step(e_id)
        v_q = self.pi_iq.step(e_iq)

        if self.v_limit > 0.0:
            mag = math.hypot(v_d, v_q)
            if mag > self.v_limit and mag > 0.0:
                scale = self.v_limit / mag
                v_d *= scale
                v_q *= scale

        eps = 1e-6
        if self.Rr > 0.0 and self.Lr > eps and abs(self.id_ref) > eps:
            omega_slip = (self.Rr / self.Lr) * (i_q_ref / abs(self.id_ref))
        else:
            omega_slip = 0.0
        omega_m = float(self.motor.state.omega_m)
        self.omega_syn = self.p * omega_m + omega_slip
        self.theta_e += self.omega_syn * self.dt

        v_abc, (v_d, v_q) = self.inverter.output(v_d, v_q, self.theta_e)
        state, i_d, i_q, torque_e, omega_m = self.motor.step(
            v_d, v_q, load_torque, self.dt, omega_syn=self.omega_syn
        )

        self.last_torque = float(torque_e)
        i_abc = dq_to_abc(i_d, i_q, self.theta_e)
        self.last_currents_abc = i_abc
        p_el = float(v_abc[0] * i_abc[0] + v_abc[1] * i_abc[1] + v_abc[2] * i_abc[2])
        i_rms = float(math.sqrt((i_abc[0] ** 2 + i_abc[1] ** 2 + i_abc[2] ** 2) / 3.0))
        return {
            "omega": float(state.omega_m),
            "i_d": float(i_d),
            "i_q": float(i_q),
            "torque": float(torque_e),
            "p_el": p_el,
            "i_rms": i_rms,
        }

    def run(self, controller, scenario_name: str) -> Dict[str, np.ndarray]:
        self.reset()
        omega_ref_func, load_func = get_scenario(scenario_name, self.env)

        steps = int(max(self.t_end / self.dt, 1))
        t = np.zeros(steps, dtype=float)
        omega = np.zeros(steps, dtype=float)
        omega_ref = np.zeros(steps, dtype=float)
        omega_dot = np.zeros(steps, dtype=float)
        iq_ref = np.zeros(steps, dtype=float)
        iq_meas = np.zeros(steps, dtype=float)
        id_meas = np.zeros(steps, dtype=float)
        torque = np.zeros(steps, dtype=float)
        load_torque = np.zeros(steps, dtype=float)
        p_el = np.zeros(steps, dtype=float)
        i_rms = np.zeros(steps, dtype=float)

        omega_prev = 0.0
        controller.reset()

        for k in range(steps):
            t[k] = self.t
            omega_ref[k] = float(omega_ref_func(self.t))
            load_torque[k] = float(load_func(self.t))
            omega_now = float(self.motor.state.omega_m)
            omega_dot[k] = (omega_now - omega_prev) / self.dt
            omega_prev = omega_now

            iq_cmd = float(controller.step(omega_ref[k], omega_now, omega_dot[k]))
            if self.iq_limit > 0.0:
                iq_cmd = max(-self.iq_limit, min(self.iq_limit, iq_cmd))
            iq_ref[k] = iq_cmd

            out = self._current_loop(iq_cmd, load_torque[k])
            omega[k] = out["omega"]
            id_meas[k] = out["i_d"]
            iq_meas[k] = out["i_q"]
            torque[k] = out["torque"]
            p_el[k] = out["p_el"]
            i_rms[k] = out["i_rms"]

            self.t += self.dt
            if self.t >= self.t_end:
                break

        last = k + 1
        return {
            "t": t[:last],
            "omega": omega[:last],
            "omega_ref": omega_ref[:last],
            "omega_dot": omega_dot[:last],
            "i_q_ref": iq_ref[:last],
            "i_q": iq_meas[:last],
            "i_d": id_meas[:last],
            "torque": torque[:last],
            "load_torque": load_torque[:last],
            "p_el": p_el[:last],
            "i_rms": i_rms[:last],
        }


__all__ = ["MotorEnv"]
