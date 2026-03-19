from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class AiIdRefSupervisorConfig:
    enabled: bool = False
    speed_tol_rel: float = 0.05
    speed_tol_abs: float = 0.0
    omega_min_pu: float = 0.1
    update_steps: int = 20
    dither_amp: float = 0.04
    bias_step: float = 0.01
    bias_max: float = 0.25
    objective: str = "specific_power"  # "specific_power", "p_in", "eta_inv"
    shaft_eps: float = 10.0
    reset_decay: float = 0.98
    objective_clip: float | None = 10.0
    idle_enable: bool = False
    idle_omega_pu: float = 0.05
    idle_action: float = -1.0
    idle_blend: float = 1.0
    idle_exit_boost_steps: int = 0
    idle_exit_action: float = 1.0
    idle_bias_decay: float = 0.95


class AiIdRefSupervisor:
    """
    Online extremum-seeking supervisor around AI id_ref action.

    The supervisor adds a small dither and updates a bias term to minimize
    a power-oriented objective while speed-error gates are satisfied.
    """

    def __init__(self, cfg: AiIdRefSupervisorConfig, omega_nominal: float) -> None:
        self.cfg = cfg
        self.omega_nominal = float(max(abs(omega_nominal), 1e-6))
        self.reset()

    def reset(self) -> None:
        self.bias = 0.0
        self.phase = 1.0
        self.obj_acc = 0.0
        self.obj_count = 0
        self.obj_plus: float | None = None
        self.obj_minus: float | None = None
        self._idle_prev = False
        self._idle_exit_left = 0

    def _gate_open(self, omega_ref: float, omega: float) -> bool:
        omega_ref_abs = abs(float(omega_ref))
        if omega_ref_abs < float(self.cfg.omega_min_pu) * self.omega_nominal:
            return False
        err = abs(float(omega_ref) - float(omega))
        lim = max(
            float(self.cfg.speed_tol_abs),
            float(self.cfg.speed_tol_rel) * max(omega_ref_abs, 1e-6),
        )
        return err <= lim

    @staticmethod
    def _clip_action(value: float) -> float:
        return float(np.clip(float(value), -1.0, 1.0))

    def adjust_action(self, ai_action: float, omega_ref: float, omega: float) -> tuple[float, bool]:
        action = float(ai_action)
        if not bool(self.cfg.enabled):
            return self._clip_action(action), False

        idle_now = bool(self.cfg.idle_enable) and abs(float(omega_ref)) < float(self.cfg.idle_omega_pu) * self.omega_nominal
        if idle_now:
            self._idle_prev = True
            self._idle_exit_left = 0
            self.bias *= float(np.clip(self.cfg.idle_bias_decay, 0.0, 1.0))
            self.obj_acc = 0.0
            self.obj_count = 0
            self.obj_plus = None
            self.obj_minus = None
            self.phase = 1.0
            idle_target = min(action, float(self.cfg.idle_action))
            idle_blend = float(np.clip(self.cfg.idle_blend, 0.0, 1.0))
            idle_action = action + idle_blend * (idle_target - action)
            return self._clip_action(idle_action), False

        if self._idle_prev:
            self._idle_prev = False
            self._idle_exit_left = int(max(self.cfg.idle_exit_boost_steps, 0))
        if self._idle_exit_left > 0:
            self._idle_exit_left -= 1
            boosted = max(action, float(self.cfg.idle_exit_action))
            return self._clip_action(boosted), False

        gate = self._gate_open(omega_ref, omega)
        if not gate:
            self.bias *= float(np.clip(self.cfg.reset_decay, 0.0, 1.0))
            self.obj_acc = 0.0
            self.obj_count = 0
            self.obj_plus = None
            self.obj_minus = None
            self.phase = 1.0
            return self._clip_action(action), False

        corr = float(self.bias + self.phase * float(self.cfg.dither_amp))
        return self._clip_action(action + corr), True

    def _objective(self, p_in_pos: float, p_shaft_pos: float) -> float:
        mode = str(self.cfg.objective).lower().strip()
        pin = max(float(p_in_pos), 0.0)
        pshaft = max(float(p_shaft_pos), 0.0)
        if mode == "p_in":
            val = pin
        elif mode == "eta_inv":
            eta = pshaft / max(pin, 1e-9)
            val = 1.0 - eta
        else:
            # specific electrical power per mechanical watt.
            val = pin / max(pshaft, float(self.cfg.shaft_eps))
        clip = self.cfg.objective_clip
        if clip is not None:
            val = float(np.clip(val, -abs(float(clip)), abs(float(clip))))
        return float(val)

    def update(self, p_in_pos: float, p_shaft_pos: float, gate_open: bool) -> None:
        if not bool(self.cfg.enabled) or not bool(gate_open):
            return
        obj = self._objective(p_in_pos, p_shaft_pos)
        self.obj_acc += obj
        self.obj_count += 1
        if self.obj_count < int(max(self.cfg.update_steps, 1)):
            return

        obj_mean = self.obj_acc / max(self.obj_count, 1)
        if self.phase > 0.0:
            self.obj_plus = float(obj_mean)
        else:
            self.obj_minus = float(obj_mean)

        self.obj_acc = 0.0
        self.obj_count = 0
        self.phase *= -1.0

        if self.obj_plus is None or self.obj_minus is None:
            return

        dither = max(abs(float(self.cfg.dither_amp)), 1e-6)
        grad = (self.obj_plus - self.obj_minus) / (2.0 * dither)
        if math.isfinite(grad) and abs(grad) > 1e-12:
            self.bias -= float(self.cfg.bias_step) * math.copysign(1.0, grad)
            self.bias = float(np.clip(self.bias, -abs(float(self.cfg.bias_max)), abs(float(self.cfg.bias_max))))
        self.obj_plus = None
        self.obj_minus = None


__all__ = ["AiIdRefSupervisorConfig", "AiIdRefSupervisor"]
