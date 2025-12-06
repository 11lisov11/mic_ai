"""
Factory for building a simple direct-voltage environment from a config file.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from vfd_ai.ident.motor_params import MotorParamsTrue


class DirectVoltageEnv:
    """
    Minimal environment that applies dq voltage references directly to the motor model.
    """

    def __init__(self, env_config: Any):
        if not hasattr(env_config, "sim"):
            raise ValueError("env_config must expose 'sim' with dt")
        if not hasattr(env_config.sim, "dt"):
            raise ValueError("env_config.sim must expose dt")

        self.env_config = env_config
        self.dt = float(env_config.sim.dt)
        self.motor = InductionMotorModel(env_config.motor)
        self.inverter = IdealInverter(env_config.inverter)
        self.u_d_ref = 0.0
        self.u_q_ref = 0.0
        self.t = 0.0
        self.theta_e = 0.0  # keep synchronous frame fixed for locked-rotor style tests

        motor_cfg = env_config.motor
        # Convert leakage + magnetizing to total stator/rotor inductances where possible.
        if hasattr(motor_cfg, "Ls_sigma") and hasattr(motor_cfg, "Lr_sigma") and hasattr(motor_cfg, "Lm"):
            Ls = float(motor_cfg.Ls_sigma + motor_cfg.Lm)
            Lr = float(motor_cfg.Lr_sigma + motor_cfg.Lm)
        else:
            Ls = float(getattr(motor_cfg, "Ls", 0.0))
            Lr = float(getattr(motor_cfg, "Lr", 0.0))
        self.motor_true_params = MotorParamsTrue(
            Rs=float(getattr(motor_cfg, "Rs", 0.0)),
            Rr=float(getattr(motor_cfg, "Rr", 0.0)),
            Ls=Ls,
            Lr=Lr,
            Lm=float(getattr(motor_cfg, "Lm", 0.0)),
            J=float(getattr(motor_cfg, "J", 0.0)),
            B=float(getattr(motor_cfg, "B", 0.0)),
        )

        self.i_d = 0.0
        self.i_q = 0.0
        self.w_mech = 0.0

    def reset(self):
        self.motor = InductionMotorModel(self.env_config.motor)
        self.u_d_ref = 0.0
        self.u_q_ref = 0.0
        self.t = 0.0
        self.theta_e = 0.0
        self.i_d = 0.0
        self.i_q = 0.0
        self.w_mech = 0.0
        return asdict(self.motor.state)

    def set_voltage_dq(self, u_d: float, u_q: float) -> None:
        self.u_d_ref = float(u_d)
        self.u_q_ref = float(u_q)

    def step(self, u_d: float | None = None, u_q: float | None = None):
        if u_d is not None or u_q is not None:
            self.u_d_ref = float(self.u_d_ref if u_d is None else u_d)
            self.u_q_ref = float(self.u_q_ref if u_q is None else u_q)

        v_abc, (v_d, v_q) = self.inverter.output(self.u_d_ref, self.u_q_ref, self.theta_e)
        _, i_d, i_q, _, omega_m = self.motor.step(
            v_d, v_q, load_torque=0.0, dt=self.dt, omega_syn=0.0
        )

        self.i_d = float(i_d)
        self.i_q = float(i_q)
        self.w_mech = float(omega_m)
        self.t += self.dt
        return v_abc, (v_d, v_q)


def _load_config_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("vfd_ai_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure module is discoverable during execution (needed for dataclass/typing checks).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_env_config(config_module: types.ModuleType) -> Any:
    if hasattr(config_module, "ENV"):
        return getattr(config_module, "ENV")
    raise AttributeError("Config module must define ENV object")


def make_env_from_config(config_path: str) -> DirectVoltageEnv:
    """
    Load a config file (python module with ENV) and return a DirectVoltageEnv.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")
    config_module = _load_config_module(path)
    env_config = _resolve_env_config(config_module)
    env = DirectVoltageEnv(env_config)

    # Propagate optional identification hints from config module to env instance
    for name in (
        "ident_u_d_step",
        "ident_total_time",
        "ident_u_q_step",
        "ident_locked_total_time",
        "ident_torque_ref",
        "ident_runup_time",
        "ident_coast_time",
    ):
        if hasattr(config_module, name):
            setattr(env, name, getattr(config_module, name))

    return env


__all__ = ["DirectVoltageEnv", "make_env_from_config"]
