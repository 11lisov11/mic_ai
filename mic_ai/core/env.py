"""
Фабрика для создания простого окружения с прямой подачей dq-напряжений по конфигурационному файлу.
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
from mic_ai.ident.motor_params import MotorParamsTrue


class DirectVoltageEnv:
    """
    Минимальная среда, которая напрямую подаёт ссылочные dq-напряжения в модель двигателя.
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
        self.theta_e = 0.0  # фиксируем синхронный кадр для locked-rotor тестов

        motor_cfg = env_config.motor
        # Конвертируем рассеяние и намагничивание в полные индуктивности статора/ротора, где это возможно.
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
        self.torque_e = 0.0
        self.load_torque = 0.0
        self._rotor_locked = False

    def reset(self):
        self.motor = InductionMotorModel(self.env_config.motor)
        self.u_d_ref = 0.0
        self.u_q_ref = 0.0
        self.t = 0.0
        self.theta_e = 0.0
        self.i_d = 0.0
        self.i_q = 0.0
        self.w_mech = 0.0
        self.torque_e = 0.0
        self.load_torque = 0.0
        self._rotor_locked = False
        return asdict(self.motor.state)

    def set_voltage_dq(self, u_d: float, u_q: float) -> None:
        self.u_d_ref = float(u_d)
        self.u_q_ref = float(u_q)

    def set_torque_command(self, torque_cmd: float) -> None:
        # Open-loop approximation: map torque command to q-axis voltage.
        self.set_voltage_dq(0.0, float(torque_cmd))

    def set_load_torque(self, torque: float) -> None:
        self.load_torque = float(torque)

    def lock_rotor(self, enabled: bool = True) -> None:
        self._rotor_locked = bool(enabled)
        if self._rotor_locked:
            self.motor.state.omega_m = 0.0

    def read_currents_dq(self) -> tuple[float, float]:
        return float(self.i_d), float(self.i_q)

    def read_mech_speed(self) -> float:
        return float(self.w_mech)

    def read_torque(self) -> float:
        return float(self.torque_e)

    def step(self, u_d: float | None = None, u_q: float | None = None):
        if u_d is not None or u_q is not None:
            self.u_d_ref = float(self.u_d_ref if u_d is None else u_d)
            self.u_q_ref = float(self.u_q_ref if u_q is None else u_q)

        v_abc, (v_d, v_q) = self.inverter.output(self.u_d_ref, self.u_q_ref, self.theta_e)
        if self._rotor_locked:
            self.motor.state.omega_m = 0.0
        _, i_d, i_q, torque_e, omega_m = self.motor.step(
            v_d, v_q, load_torque=self.load_torque, dt=self.dt, omega_syn=0.0
        )
        if self._rotor_locked:
            self.motor.state.omega_m = 0.0
            omega_m = 0.0

        self.i_d = float(i_d)
        self.i_q = float(i_q)
        self.w_mech = float(omega_m)
        self.torque_e = float(torque_e)
        self.t += self.dt
        return v_abc, (v_d, v_q)


def _load_config_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("mic_ai_user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Делаем модуль доступным во время исполнения (нужно для dataclass/typing проверок).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_env_config(config_module: types.ModuleType) -> Any:
    if hasattr(config_module, "ENV"):
        return getattr(config_module, "ENV")
    raise AttributeError("Config module must define ENV object")


def make_env_from_config(config_path: str) -> DirectVoltageEnv:
    """
    Загрузить конфиг (python-модуль с ENV) и вернуть DirectVoltageEnv.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")
    config_module = _load_config_module(path)
    env_config = _resolve_env_config(config_module)
    env = DirectVoltageEnv(env_config)

    # Протягиваем необязательные подсказки для идентификации из конфигурационного модуля в экземпляр env
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
