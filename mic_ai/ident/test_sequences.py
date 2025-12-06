"""
Генерация тестовых профилей для идентификации.
"""

from __future__ import annotations

import numpy as np


def make_rs_leq_test_profile(dt: float, total_time: float, u_d_step: float) -> dict:
    """
    Построить простой ступенчатый профиль напряжения по d-оси для оценки Rs/Leq.

    Возвращает словарь с временной сеткой и ссылками по напряжению.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if total_time <= 0:
        raise ValueError("total_time must be positive")

    n_steps = int(np.floor(total_time / dt))
    if n_steps < 2:
        raise ValueError("total_time must be at least two integration steps")

    t = np.arange(n_steps, dtype=float) * dt
    u_d_ref = np.zeros_like(t)
    u_q_ref = np.zeros_like(t)

    warmup_steps = min(max(10, int(0.02 * n_steps)), n_steps - 1)
    u_d_ref[warmup_steps:] = u_d_step
    # u_q_ref оставляем равным нулю

    return {"t": t, "u_d_ref": u_d_ref, "u_q_ref": u_q_ref}


__all__ = ["make_rs_leq_test_profile"]
