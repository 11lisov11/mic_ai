"""
Идеальный инвертор напряжения с ограничением по амплитуде.
"""

from __future__ import annotations

import math
from typing import Tuple

from config.env import InverterParams
from models.transformations import abc_to_dq, dq_to_abc


class IdealInverter:
    def __init__(self, params: InverterParams):
        self.params = params

    def output(
        self, v_d: float, v_q: float, theta_e: float, i_abc: Tuple[float, float, float] | None = None
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

        if i_abc is not None:
            try:
                i_vals = (float(i_abc[0]), float(i_abc[1]), float(i_abc[2]))
            except Exception:
                i_vals = (0.0, 0.0, 0.0)

            r_out = float(getattr(self.params, "r_out", 0.0) or 0.0)
            dead_time = float(getattr(self.params, "dead_time", 0.0) or 0.0)
            v_drop = float(getattr(self.params, "v_drop", 0.0) or 0.0)
            v_dt = 0.0
            if dead_time > 0.0 and self.params.f_pwm > 0.0:
                v_dt = float(self.params.Vdc * dead_time * self.params.f_pwm)

            def _sign(x: float) -> float:
                return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)

            if r_out != 0.0 or v_dt != 0.0 or v_drop != 0.0:
                v_abc = tuple(
                    float(v - (v_dt + v_drop) * _sign(i) - r_out * i)
                    for v, i in zip(v_abc, i_vals)
                )
                v_d, v_q = abc_to_dq(*v_abc, theta_e)
                v_mag = math.sqrt(v_d * v_d + v_q * v_q)
                if v_mag > v_max and v_mag > 0.0:
                    scale = v_max / v_mag
                    v_d *= scale
                    v_q *= scale
                    v_abc = dq_to_abc(v_d, v_q, theta_e)
        return v_abc, (v_d, v_q)


__all__ = ["IdealInverter"]
