# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict

import numpy as np


def _step_dt(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(np.mean(np.diff(t)))


def calc_metrics(
    t: np.ndarray,
    omega: np.ndarray,
    omega_ref: np.ndarray,
    i_q: np.ndarray | None = None,
    i_rms: np.ndarray | None = None,
    iq_limit: float | None = None,
    speed_tol_rel: float = 0.05,
    speed_tol_abs: float | None = None,
) -> Dict[str, float]:
    dt = _step_dt(t)
    if dt <= 0.0:
        dt = 1.0

    err = omega_ref - omega
    j_speed = float(np.mean(np.abs(err))) if err.size else 0.0

    if i_q is not None:
        energy_signal = i_q
    elif i_rms is not None:
        energy_signal = i_rms
    else:
        energy_signal = np.zeros_like(omega)
    j_energy = float(np.sum(energy_signal * energy_signal) * dt) if energy_signal.size else 0.0

    if speed_tol_abs is None:
        speed_lim = speed_tol_rel * np.maximum(np.abs(omega_ref), 1e-6)
    else:
        speed_lim = float(speed_tol_abs)
    speed_viol = np.abs(err) > speed_lim

    if iq_limit is None:
        current_viol = np.zeros_like(speed_viol, dtype=bool)
    else:
        current_viol = np.abs(energy_signal) > float(iq_limit)
    j_stability = float(np.mean(speed_viol | current_viol)) if err.size else 0.0

    return {
        "J_speed": j_speed,
        "J_energy": j_energy,
        "J_stability": j_stability,
    }


def weighted_score(metrics: Dict[str, float], w_speed: float, w_energy: float, w_stability: float) -> float:
    return float(
        w_speed * metrics.get("J_speed", 0.0)
        + w_energy * metrics.get("J_energy", 0.0)
        + w_stability * metrics.get("J_stability", 0.0)
    )
