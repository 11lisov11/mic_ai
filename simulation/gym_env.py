"""
Gym-окружение, которое объединяет модель двигателя, инвертор и контроллеры.
"""

from __future__ import annotations

import math
import types
from typing import Any, Dict, Optional, Tuple
from dataclasses import replace

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:  # минимальный запасной вариант, если gym не установлен
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.array(low if shape is None else np.full(shape, low), dtype=dtype)
            self.high = np.array(high if shape is None else np.full(shape, high), dtype=dtype)
            self.shape = shape if shape is not None else self.low.shape
            self.dtype = dtype

        def contains(self, x) -> bool:
            arr = np.asarray(x, dtype=self.dtype)
            return arr.shape == tuple(self.shape) and np.all(arr >= self.low) and np.all(arr <= self.high)

    class _Env:
        metadata: Dict[str, Any] = {}

        def __init__(self):
            ...

    gym = types.SimpleNamespace(Env=_Env)
    spaces = types.SimpleNamespace(Box=_Box)

from config.env import ENV, EnvConfig
from control.scalar_vf import ScalarVfController
from control.vector_foc import FocController
from control.v3_ternary import V3Controller
from control.hybrid_v3_foc import HybridParams, HybridV3FocController
from control.id_ref_lut import IdRefLut
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import dq_to_abc
from simulation.scenarios import get_scenario


class InductionMotorEnv(gym.Env):
    """
    Gym-совместимая среда для управления асинхронным двигателем в режимах V/f или FOC.
    """

    metadata = {"render.modes": []}

    def __init__(self, env_config: EnvConfig = ENV):
        super().__init__()
        self.env = env_config
        self.dt = env_config.sim.dt
        self.mode = env_config.sim.mode.lower()
        self.sigma_omega = max(float(getattr(env_config.sim, "sigma_omega", 0.0) or 0.0), 0.0)
        self.sigma_i_abc = max(float(getattr(env_config.sim, "sigma_i_abc", 0.0) or 0.0), 0.0)
        self.loss_inv_r = float(getattr(env_config, "loss_inv_r", 0.0) or 0.0)
        self.loss_core_k = float(getattr(env_config, "loss_core_k", 0.0) or 0.0)
        self.loss_core_omega_exp = float(getattr(env_config, "loss_core_omega_exp", 1.0) or 1.0)
        self.loss_core_psi_exp = float(getattr(env_config, "loss_core_psi_exp", 2.0) or 2.0)

        self.motor = InductionMotorModel(env_config.motor)
        self.inverter = IdealInverter(env_config.inverter)

        if self.mode == "scalar":
            self.controller = ScalarVfController(
                env_config.scalar_vf, self.dt, env_config.motor.p, env_config.inverter.Vdc
            )
        elif self.mode == "foc":
            self.controller = FocController(env_config.foc, env_config.motor, self.dt)
        elif self.mode == "v3":
            omega_base = 2.0 * math.pi * env_config.scalar_vf.f_max / env_config.motor.p
            self.controller = V3Controller(env_config.foc, env_config.motor, self.dt, omega_base)
        elif self.mode == "hybrid":
            omega_base = 2.0 * math.pi * env_config.scalar_vf.f_max / env_config.motor.p
            hy_params = HybridParams(
                err_tol_rel=float(getattr(env_config, "hybrid_err_tol_rel", 0.02)),
                err_tol_abs=float(getattr(env_config, "hybrid_err_tol_abs", 0.0)),
                min_omega_pu=float(getattr(env_config, "hybrid_min_omega_pu", 0.1)),
                load_low_ratio=float(getattr(env_config, "hybrid_load_low_ratio", 0.6)),
            )
            self.controller = HybridV3FocController(
                env_config.foc,
                env_config.motor,
                self.dt,
                omega_base,
                load_nom=float(max(abs(env_config.sim.load_torque), 1e-6)),
                hybrid_params=hy_params,
            )
        else:
            raise ValueError(f"Unknown control mode '{self.mode}'")

        self._id_ref_lut = None
        lut_path = getattr(env_config, "id_ref_lut_path", None)
        if lut_path:
            try:
                self._id_ref_lut = IdRefLut.from_json(lut_path)
            except Exception:
                self._id_ref_lut = None

        self.omega_ref_func, self.load_torque_func = get_scenario(env_config.sim.scenario_name, env_config)

        # действие: нормализованная команда скорости в диапазоне [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # наблюдение: omega_m, omega_ref, T_e, i_a, i_b, i_c, P_in, P_out
        obs_low = np.array([-np.inf] * 8, dtype=np.float32)
        obs_high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.omega_base = 2.0 * math.pi * env_config.scalar_vf.f_max / env_config.motor.p
        self.theta_mech = 0.0
        self.last_currents_abc = (0.0, 0.0, 0.0)
        self.last_torque = 0.0
        self.t = 0.0

    def reset(self) -> np.ndarray:
        self.motor = InductionMotorModel(self.env.motor)
        self.controller.reset()
        self.theta_mech = 0.0
        self.last_currents_abc = (0.0, 0.0, 0.0)
        self.last_torque = 0.0
        self.t = 0.0
        self.omega_ref_func, self.load_torque_func = get_scenario(self.env.sim.scenario_name, self.env)

        obs, _, _ = self._build_observation(
            omega_ref=0.0,
            torque_e=0.0,
            i_abc=(0.0, 0.0, 0.0),
            v_abc=(0.0, 0.0, 0.0),
            omega_m=0.0,
        )
        return obs

    def _build_observation(
        self,
        omega_ref: float,
        torque_e: float,
        i_abc: Tuple[float, float, float],
        v_abc: Tuple[float, float, float],
        omega_m: float,
    ) -> Tuple[np.ndarray, float, float]:
        p_in = v_abc[0] * i_abc[0] + v_abc[1] * i_abc[1] + v_abc[2] * i_abc[2]
        p_out = torque_e * omega_m
        obs = np.array(
            [omega_m, omega_ref, torque_e, i_abc[0], i_abc[1], i_abc[2], p_in, p_out],
            dtype=np.float32,
        )
        return obs, p_in, p_out

    def _apply_action(self, action: Optional[np.ndarray], omega_ref_scenario: float) -> float:
        if action is None:
            return omega_ref_scenario
        value = float(np.asarray(action).flatten()[0])
        value = max(-1.0, min(1.0, value))
        return value * self.omega_base

    def step(self, action: Optional[np.ndarray] = None):
        t = self.t
        omega_ref = self.omega_ref_func(t)
        load_torque = self.load_torque_func(t)

        if self._id_ref_lut is not None and self.mode == "foc":
            try:
                id_ref = float(self._id_ref_lut.query(omega_ref, load_torque))
                self.controller.params = replace(self.controller.params, id_ref=id_ref)
            except Exception:
                pass

        omega_ref = self._apply_action(action, omega_ref)
        
        # --- Унифицированный шаг управления ---
        # Контроллер ожидает i_abc с предыдущего шага (или отфильтрованные)
        omega_m_true = self.motor.state.omega_m
        i_abc_true = self.last_currents_abc
        omega_m_meas = omega_m_true + np.random.randn() * self.sigma_omega if self.sigma_omega > 0 else omega_m_true
        if self.sigma_i_abc > 0:
            i_abc_meas = tuple(float(x + np.random.randn() * self.sigma_i_abc) for x in i_abc_true)
        else:
            i_abc_meas = i_abc_true

        if self.mode == "hybrid":
            v_d, v_q, theta_e, omega_syn, ctrl_info = self.controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=self.last_torque,
                theta_mech=self.theta_mech,
                load_torque=load_torque,
            )
        else:
            v_d, v_q, theta_e, omega_syn, ctrl_info = self.controller.step(
                t=t,
                omega_ref=omega_ref,
                omega_m=omega_m_meas,
                i_abc=i_abc_meas,
                torque_e=self.last_torque,
                theta_mech=self.theta_mech
            )
        
        # Обновление инвертора и двигателя
        v_abc, (v_d, v_q) = self.inverter.output(v_d, v_q, theta_e, self.last_currents_abc)
        state, i_d, i_q, torque_e, omega_m = self.motor.step(
            v_d, v_q, load_torque, self.dt, omega_syn=omega_syn
        )
        
        self.theta_mech += omega_m * self.dt
        i_abc = dq_to_abc(i_d, i_q, theta_e)
        
        self.last_torque = torque_e
        self.last_currents_abc = i_abc

        obs, p_in, p_out = self._build_observation(omega_ref, torque_e, i_abc, v_abc, state.omega_m)

        # Loss-aware power accounting (optional; defaults to zero extras).
        try:
            i_rms = math.sqrt((i_abc[0] ** 2 + i_abc[1] ** 2 + i_abc[2] ** 2) / 3.0)
        except Exception:
            i_rms = 0.0
        p_inv = 3.0 * self.loss_inv_r * (i_rms ** 2) if self.loss_inv_r > 0.0 else 0.0
        p_core = 0.0
        if self.loss_core_k > 0.0:
            psi_s = math.hypot(state.psi_ds, state.psi_qs)
            omega_core = abs(omega_syn)
            p_core = float(self.loss_core_k) * (omega_core ** self.loss_core_omega_exp) * (psi_s ** self.loss_core_psi_exp)
        p_in_total = p_in + p_inv + p_core
        p_mech_loss = float(self.motor.params.B) * (omega_m ** 2)
        try:
            _, _, i_dr, i_qr = self.motor._currents(state)
        except Exception:
            i_dr, i_qr = 0.0, 0.0

        self.t += self.dt
        done = self.t >= self.env.sim.t_end
        
        info: Dict[str, Any] = {
            "omega_ref": omega_ref,
            "torque_e": torque_e,
            "p_in": p_in,
            "p_out": p_out,
            "p_in_total": p_in_total,
            "p_in_total_pos": max(0.0, p_in_total),
            "p_inv_loss": p_inv,
            "p_core_loss": p_core,
            "p_mech_loss": p_mech_loss,
            "i_rms": i_rms,
            "i_dr": float(i_dr),
            "i_qr": float(i_qr),
            "i_abc": i_abc,
            "v_abc": v_abc,
            "theta_e": theta_e,
            "omega_syn": omega_syn,
        }
        # добавляем отладочную информацию контроллера
        info.update(ctrl_info)

        reward = 0.0
        return obs, reward, done, info


__all__ = ["InductionMotorEnv"]


__all__ = ["InductionMotorEnv"]
