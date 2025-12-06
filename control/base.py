from __future__ import annotations
from typing import Protocol, Tuple

class BaseController(Protocol):
    """
    Протокол для контроллеров двигателя.
    """
    
    def reset(self) -> None:
        """Сбросить внутреннее состояние."""
        ...

    def step(
        self,
        t: float,
        omega_ref: float,
        omega_m: float,
        i_abc: Tuple[float, float, float],
        torque_e: float,
        theta_mech: float
    ) -> Tuple[float, float, float, float]:
        """
        Вычислить команды напряжений в dq.

        Args:
            t: текущее время симуляции.
            omega_ref: заданная механическая скорость (рад/с).
            omega_m: измеренная механическая скорость (рад/с).
            i_abc: измеренные фазные токи (А).
            torque_e: измеренный/оценённый момент (Нм).
            theta_mech: механический угол (рад).

        Returns:
            v_d: команда напряжения по d-оси (В).
            v_q: команда напряжения по q-оси (В).
            theta_e: электрический угол для преобразования координат (рад).
            omega_syn: синхронная электрическая скорость (рад/с).
        """
        ...
