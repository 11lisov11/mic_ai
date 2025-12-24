from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def calc_i_rms(i_abc: Iterable[float]) -> float:
    values = np.asarray(i_abc, dtype=float)
    if values.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(values * values)))


def calc_p_el(v_abc: Iterable[float], i_abc: Iterable[float]) -> float:
    v_vals = np.asarray(v_abc, dtype=float)
    i_vals = np.asarray(i_abc, dtype=float)
    if v_vals.size == 0 or i_vals.size == 0:
        return 0.0
    return float(np.sum(v_vals * i_vals))


def calc_p_mech(omega: float, torque: float) -> float:
    return float(omega * torque)
