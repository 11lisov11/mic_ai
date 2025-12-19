"""
Упрощённая dq-модель короткозамкнутого асинхронного двигателя.
"""

from __future__ import annotations

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

    def update_params(self, params: MotorParams) -> None:
        """Update motor parameters and refresh derived inductances."""
        self.params = params
        self.Ls = params.Ls_sigma + params.Lm
        self.Lr = params.Lr_sigma + params.Lm
        self.denom = self.Ls * self.Lr - params.Lm ** 2

    def _currents(self, state: MotorState) -> Tuple[float, float, float, float]:
        """
        Рассчитать dq-токи статора и ротора по потокосцеплениям.
        """
        i_ds = (state.psi_ds * self.Lr - state.psi_dr * self.params.Lm) / self.denom
        i_qs = (state.psi_qs * self.Lr - state.psi_qr * self.params.Lm) / self.denom
        i_dr = (state.psi_dr * self.Ls - state.psi_ds * self.params.Lm) / self.denom
        i_qr = (state.psi_qr * self.Ls - state.psi_qs * self.params.Lm) / self.denom
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
