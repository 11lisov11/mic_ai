from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from config.env import EnvConfig, create_default_env
from safety.supervisor import SafetySupervisor, SafetyLimits

from .base_driver import BaseDriver


class HwDriverStub(BaseDriver):
    """Hardware driver stub with simple synthetic telemetry."""

    def __init__(self, env_config: EnvConfig | None = None) -> None:
        self._env = env_config or create_default_env()
        self._dt = float(self._env.sim.dt)
        self._mode = "FOC"
        self._rng = np.random.default_rng()

        self._omega = 0.0
        self._omega_ref = 0.0
        self._theta_e = 0.0
        self._torque = 0.0
        self._load_torque = 0.0
        self._i_d = 0.0
        self._i_q = 0.0
        self._t = 0.0

        self._pending_action = (0.0, 0.0)
        self._last_fault: Optional[str] = None
        self._stopped = False

        self._pi_iq = 0.0
        self._pi_kp = 0.5
        self._pi_ki = 5.0
        self._tau = 0.15
        self._k_vq = 0.2
        self._k_load = 0.05

        v_nom = self._env.inverter.Vdc / math.sqrt(3.0) if self._env.inverter.Vdc > 0.0 else None
        i_nom = float(getattr(self._env.foc, "iq_limit", 0.0) or 0.0)
        if i_nom <= 0.0:
            i_nom = float(getattr(self._env.motor, "I_n", 0.0) or 0.0)
        omega_base = 2.0 * math.pi * self._env.scalar_vf.f_max / self._env.motor.p

        self._safety = SafetySupervisor(self._dt)
        self._safety.configure(
            SafetyLimits(
                i_max=i_nom if i_nom > 0.0 else None,
                v_max=v_nom if v_nom is not None else None,
                omega_max=1.5 * omega_base if omega_base > 0.0 else None,
            )
        )
        self._last_obs: Dict[str, float] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._omega = 0.0
        self._omega_ref = 0.0
        self._theta_e = 0.0
        self._torque = 0.0
        self._load_torque = 0.0
        self._i_d = 0.0
        self._i_q = 0.0
        self._t = 0.0
        self._pending_action = (0.0, 0.0)
        self._last_fault = None
        self._stopped = False
        self._pi_iq = 0.0
        self._safety.reset()
        self._last_obs = self._build_obs()

    def set_mode(self, mode: str) -> None:
        mode_upper = str(mode).upper()
        if mode_upper not in ("FOC", "MIC"):
            raise ValueError(f"Unknown mode '{mode}' (expected 'FOC' or 'MIC').")
        self._mode = mode_upper

    def set_limits(self, limits: Dict[str, float]) -> None:
        self._safety.configure(limits)

    def apply_action(self, vd: float, vq: float) -> None:
        if self._stopped:
            return
        self._pending_action = (float(vd), float(vq))

    def step(self) -> None:
        if self._stopped:
            return

        # Fake reference: slow drift to a nominal speed to emulate scenario dependence.
        if self._omega_ref == 0.0:
            omega_base = 2.0 * math.pi * self._env.scalar_vf.f_max / self._env.motor.p
            self._omega_ref = 0.8 * omega_base

        if self._mode == "FOC":
            error = self._omega_ref - self._omega
            self._pi_iq += error * self._pi_ki * self._dt
            vq_cmd = self._pi_kp * error + self._pi_iq
            vd_cmd = 0.0
        else:
            vd_cmd, vq_cmd = self._pending_action

        vd_cmd, vq_cmd = self._safety.sanitize_action((vd_cmd, vq_cmd))
        if self._safety.last_fault() is not None:
            self._stopped = True
            self._last_fault = self._safety.last_fault()
            return

        accel = (self._omega_ref - self._omega) / max(self._tau, 1e-6)
        accel += self._k_vq * vq_cmd - self._k_load * self._load_torque
        self._omega += accel * self._dt

        self._torque = self._k_vq * vq_cmd
        self._i_q = 0.5 * vq_cmd
        self._i_d = 0.1 * vd_cmd
        self._theta_e += self._omega * self._dt
        self._t += self._dt

        obs = self._build_obs()
        aborted, reason = self._safety.check_abort(obs)
        if aborted:
            self._stopped = True
            self._last_fault = reason
        self._last_obs = obs

    def read_obs(self) -> Dict[str, float]:
        return dict(self._last_obs)

    def get_last_fault(self) -> Optional[str]:
        if self._last_fault is not None:
            return self._last_fault
        return self._safety.last_fault()

    def close(self) -> None:
        return None

    def _build_obs(self) -> Dict[str, float]:
        v_d, v_q = self._pending_action
        return {
            "t": float(self._t),
            "omega": float(self._omega),
            "omega_ref": float(self._omega_ref),
            "omega_syn": float(self._omega),
            "id": float(self._i_d),
            "iq": float(self._i_q),
            "ia": float(self._i_d),
            "ib": float(self._i_q),
            "ic": float(-(self._i_d + self._i_q)),
            "u_dc": float(self._env.inverter.Vdc),
            "theta_e": float(self._theta_e),
            "torque": float(self._torque),
            "load_torque": float(self._load_torque),
            "v_d": float(v_d),
            "v_q": float(v_q),
            "v_a": float(v_d),
            "v_b": float(v_q),
            "v_c": float(-(v_d + v_q)),
            "flags": {
                "stopped": bool(self._stopped),
                "fault": self.get_last_fault(),
            },
        }


__all__ = ["HwDriverStub"]
