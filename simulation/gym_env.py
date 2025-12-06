"""
Gym environment that wraps the induction motor, inverter and controllers.
"""

from __future__ import annotations

import math
import types
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except ImportError:  # minimal fallback when gym is not installed
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
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import dq_to_abc
from simulation.scenarios import get_scenario


class InductionMotorEnv(gym.Env):
    """
    Gym-style environment for scalar V/f or FOC controlled induction motor.
    """

    metadata = {"render.modes": []}

    def __init__(self, env_config: EnvConfig = ENV):
        super().__init__()
        self.env = env_config
        self.dt = env_config.sim.dt
        self.mode = env_config.sim.mode.lower()

        self.motor = InductionMotorModel(env_config.motor)
        self.inverter = IdealInverter(env_config.inverter)

        if self.mode == "scalar":
            self.controller = ScalarVfController(
                env_config.scalar_vf, self.dt, env_config.motor.p, env_config.inverter.Vdc
            )
        elif self.mode == "foc":
            self.controller = FocController(env_config.foc, env_config.motor, self.dt)
        else:
            raise ValueError(f"Unknown control mode '{self.mode}'")

        self.omega_ref_func, self.load_torque_func = get_scenario(env_config.sim.scenario_name, env_config)

        # action: normalized speed command in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # observation: omega_m, omega_ref, T_e, i_a, i_b, i_c, P_in, P_out
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

        omega_ref = self._apply_action(action, omega_ref)
        
        # --- Unified Control Step ---
        # Note: controller expects i_abc from previous step (or filtered)
        v_d, v_q, theta_e, omega_syn, ctrl_info = self.controller.step(
            t=t,
            omega_ref=omega_ref,
            omega_m=self.motor.state.omega_m,
            i_abc=self.last_currents_abc,
            torque_e=self.last_torque,
            theta_mech=self.theta_mech
        )
        
        # Inverter and Motor Update
        v_abc, (v_d, v_q) = self.inverter.output(v_d, v_q, theta_e)
        state, i_d, i_q, torque_e, omega_m = self.motor.step(
            v_d, v_q, load_torque, self.dt, omega_syn=omega_syn
        )
        
        self.theta_mech += omega_m * self.dt
        i_abc = dq_to_abc(i_d, i_q, theta_e)
        
        self.last_torque = torque_e
        self.last_currents_abc = i_abc
        
        obs, p_in, p_out = self._build_observation(omega_ref, torque_e, i_abc, v_abc, state.omega_m)

        self.t += self.dt
        done = self.t >= self.env.sim.t_end
        
        info: Dict[str, Any] = {
            "omega_ref": omega_ref,
            "torque_e": torque_e,
            "p_in": p_in,
            "p_out": p_out,
            "i_abc": i_abc,
            "v_abc": v_abc,
            "theta_e": theta_e,
            "omega_syn": omega_syn,
        }
        # merge controller debug info
        info.update(ctrl_info)

        reward = 0.0
        return obs, reward, done, info


__all__ = ["InductionMotorEnv"]


__all__ = ["InductionMotorEnv"]
