"""
Преобразования координат между системами abc, альфа-бета и dq.
"""

from __future__ import annotations

import math
from typing import Tuple


SQRT3 = math.sqrt(3.0)
TWO_THIRDS = 2.0 / 3.0


def abc_to_alpha_beta(i_a: float, i_b: float, i_c: float) -> Tuple[float, float]:
    """Преобразование Кларка из трёхфазной системы в стационарную альфа-бета."""
    i_alpha = TWO_THIRDS * (i_a - 0.5 * (i_b + i_c))
    i_beta = TWO_THIRDS * (0.5 * SQRT3) * (i_b - i_c)
    return i_alpha, i_beta


def alpha_beta_to_dq(i_alpha: float, i_beta: float, theta_e: float) -> Tuple[float, float]:
    """Преобразование Парка из альфа-бета в вращающуюся dq-систему."""
    cos_t = math.cos(theta_e)
    sin_t = math.sin(theta_e)
    i_d = i_alpha * cos_t + i_beta * sin_t
    i_q = -i_alpha * sin_t + i_beta * cos_t
    return i_d, i_q


def dq_to_alpha_beta(v_d: float, v_q: float, theta_e: float) -> Tuple[float, float]:
    """Обратное преобразование Парка: из dq в альфа-бета."""
    cos_t = math.cos(theta_e)
    sin_t = math.sin(theta_e)
    v_alpha = v_d * cos_t - v_q * sin_t
    v_beta = v_d * sin_t + v_q * cos_t
    return v_alpha, v_beta


def alpha_beta_to_abc(v_alpha: float, v_beta: float) -> Tuple[float, float, float]:
    """Обратное преобразование Кларка из альфа-бета в трёхфазную систему."""
    v_a = v_alpha
    v_b = -0.5 * v_alpha + (SQRT3 / 2.0) * v_beta
    v_c = -0.5 * v_alpha - (SQRT3 / 2.0) * v_beta
    return v_a, v_b, v_c


def abc_to_dq(i_a: float, i_b: float, i_c: float, theta_e: float) -> Tuple[float, float]:
    """Прямое преобразование из abc в dq."""
    i_alpha, i_beta = abc_to_alpha_beta(i_a, i_b, i_c)
    return alpha_beta_to_dq(i_alpha, i_beta, theta_e)


def dq_to_abc(v_d: float, v_q: float, theta_e: float) -> Tuple[float, float, float]:
    """Прямое преобразование из dq в abc."""
    v_alpha, v_beta = dq_to_alpha_beta(v_d, v_q, theta_e)
    return alpha_beta_to_abc(v_alpha, v_beta)


__all__ = [
    "abc_to_alpha_beta",
    "alpha_beta_to_dq",
    "dq_to_alpha_beta",
    "alpha_beta_to_abc",
    "abc_to_dq",
    "dq_to_abc",
]
