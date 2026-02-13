"""
Упрощённая dq-модель короткозамкнутого асинхронного двигателя.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from config.env import MotorParams


@dataclass
class MotorState:
    psi_ds: float = 0.0
    psi_qs: float = 0.0
    psi_dr: float = 0.0
    psi_qr: float = 0.0
    omega_m: float = 0.0


class InductionMotorModel:
    """dq-модель асинхронного двигателя с шагом Эйлера."""

    def __init__(self, params: MotorParams):
        self.params = params
        self.state = MotorState()

        # предварительный расчёт индуктивностей для вычисления токов
        self.Ls = params.Ls_sigma + params.Lm
        self.Lr = params.Lr_sigma + params.Lm
        self.denom = self.Ls * self.Lr - params.Lm ** 2
        self.psi_sat = float(getattr(params, "psi_sat", 0.0) or 0.0)
        self.sat_exp = float(getattr(params, "sat_exp", 2.0) or 2.0)
        self.lm_min_scale = float(getattr(params, "lm_min_scale", 0.2) or 0.0)
        self._sat_enabled = self.psi_sat > 0.0 and self.lm_min_scale >= 0.0

    def update_params(self, params: MotorParams) -> None:
        """Update motor parameters and refresh derived inductances."""
        self.params = params
        self.Ls = params.Ls_sigma + params.Lm
        self.Lr = params.Lr_sigma + params.Lm
        self.denom = self.Ls * self.Lr - params.Lm ** 2
        self.psi_sat = float(getattr(params, "psi_sat", 0.0) or 0.0)
        self.sat_exp = float(getattr(params, "sat_exp", 2.0) or 2.0)
        self.lm_min_scale = float(getattr(params, "lm_min_scale", 0.2) or 0.0)
        self._sat_enabled = self.psi_sat > 0.0 and self.lm_min_scale >= 0.0

    def _lm_effective(self, state: MotorState) -> float:
        if not self._sat_enabled:
            return float(self.params.Lm)
        psi_s = math.hypot(state.psi_ds, state.psi_qs)
        if psi_s <= 0.0:
            return float(self.params.Lm)
        scale = 1.0 / (1.0 + (psi_s / self.psi_sat) ** self.sat_exp)
        scale = max(scale, float(self.lm_min_scale))
        return float(self.params.Lm) * scale

    def _currents(self, state: MotorState) -> Tuple[float, float, float, float]:
        """
        Рассчитать dq-токи статора и ротора по потокосцеплениям.
        """
        lm_eff = self._lm_effective(state)
        if lm_eff == float(self.params.Lm):
            Ls = self.Ls
            Lr = self.Lr
            denom = self.denom
        else:
            Ls = self.params.Ls_sigma + lm_eff
            Lr = self.params.Lr_sigma + lm_eff
            denom = Ls * Lr - lm_eff ** 2
        if denom == 0.0:
            denom = 1e-9
        i_ds = (state.psi_ds * Lr - state.psi_dr * lm_eff) / denom
        i_qs = (state.psi_qs * Lr - state.psi_qr * lm_eff) / denom
        i_dr = (state.psi_dr * Ls - state.psi_ds * lm_eff) / denom
        i_qr = (state.psi_qr * Ls - state.psi_qs * lm_eff) / denom
        return i_ds, i_qs, i_dr, i_qr

    def step(
        self,
        v_ds: float,
        v_qs: float,
        load_torque: float,
        dt: float,
        omega_syn: float | None = None,
    ) -> tuple[MotorState, float, float, float, float]:
        """
        Обновить состояние двигателя на один шаг методом прямого Эйлера.

        Args:
            v_ds: статоровое d-напряжение в синхронной системе.
            v_qs: статоровое q-напряжение в синхронной системе.
            load_torque: внешняя нагрузка по моменту.
            dt: шаг моделирования.
            omega_syn: синхронная электрическая скорость dq-кадра (рад/с).

        Returns:
            state: обновлённое состояние MotorState.
            i_ds, i_qs: dq-токи статора.
            T_e: электромагнитный момент.
            omega_m: механическая скорость (рад/с).
        """
        p = self.params
        state = self.state

        omega_m = state.omega_m
        omega_syn = omega_syn if omega_syn is not None else p.p * omega_m
        omega_r = p.p * omega_m
        omega_slip = omega_syn - omega_r

        i_ds, i_qs, i_dr, i_qr = self._currents(state)

        dpsi_ds = v_ds - p.Rs * i_ds + omega_syn * state.psi_qs
        dpsi_qs = v_qs - p.Rs * i_qs - omega_syn * state.psi_ds
        dpsi_dr = -p.Rr * i_dr + omega_slip * state.psi_qr
        dpsi_qr = -p.Rr * i_qr - omega_slip * state.psi_dr

        torque_e = 1.5 * p.p * (state.psi_ds * i_qs - state.psi_qs * i_ds)
        domega_m = (torque_e - load_torque - p.B * omega_m) / p.J

        next_state = MotorState(
            psi_ds=state.psi_ds + dt * dpsi_ds,
            psi_qs=state.psi_qs + dt * dpsi_qs,
            psi_dr=state.psi_dr + dt * dpsi_dr,
            psi_qr=state.psi_qr + dt * dpsi_qr,
            omega_m=omega_m + dt * domega_m,
        )

        self.state = next_state
        return next_state, i_ds, i_qs, torque_e, next_state.omega_m


__all__ = ["MotorState", "InductionMotorModel"]
