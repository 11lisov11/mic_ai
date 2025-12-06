"""
Идеальный инвертор напряжения с ограничением по амплитуде.
"""

from __future__ import annotations

import math
from typing import Tuple

from config.env import InverterParams
from models.transformations import dq_to_abc


class IdealInverter:
    def __init__(self, params: InverterParams):
        self.params = params

    def output(
        self, v_d: float, v_q: float, theta_e: float
    ) -> Tuple[tuple[float, float, float], tuple[float, float]]:
        """
        Применить ограничение по модулю напряжения и вернуть фазы abc.

        Returns:
            v_abc: кортеж фазных напряжений (v_a, v_b, v_c)
            v_dq: dq-напряжения с учётом насыщения (v_d, v_q)
        """
        v_mag = math.sqrt(v_d * v_d + v_q * v_q)
        v_max = self.params.Vdc / math.sqrt(3.0)

        if v_mag > v_max and v_mag > 0.0:
            scale = v_max / v_mag
            v_d *= scale
            v_q *= scale

        v_abc = dq_to_abc(v_d, v_q, theta_e)
        return v_abc, (v_d, v_q)


__all__ = ["IdealInverter"]
