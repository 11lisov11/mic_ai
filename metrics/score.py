from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


DEFAULT_SCORE_WEIGHTS = {
    "w_speed": 1.0,
    "w_current": 0.2,
    "w_power": 0.1,
}
DEFAULT_FAULT_PENALTY = 1.0e6


def _mean_or_zero(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def compute_metrics(
    t: Iterable[float],
    omega: Iterable[float],
    omega_ref: Iterable[float],
    i_rms: Iterable[float],
    p_el: Iterable[float],
) -> Dict[str, float]:
    t_arr = np.asarray(list(t), dtype=float)
    omega_arr = np.asarray(list(omega), dtype=float)
    ref_arr = np.asarray(list(omega_ref), dtype=float)
    i_rms_arr = np.asarray(list(i_rms), dtype=float)
    p_el_arr = np.asarray(list(p_el), dtype=float)

    if t_arr.size > 1:
        dt = float(np.mean(np.diff(t_arr)))
    else:
        dt = 0.0

    err = ref_arr - omega_arr
    abs_err = np.abs(err)
    iae = float(np.sum(abs_err) * dt) if dt > 0.0 else float(np.sum(abs_err))
    ise = float(np.sum(err * err) * dt) if dt > 0.0 else float(np.sum(err * err))

    mean_abs_err = _mean_or_zero(abs_err)
    mean_i_rms = _mean_or_zero(i_rms_arr)
    rms_i_rms = float(np.sqrt(_mean_or_zero(i_rms_arr * i_rms_arr)))
    mean_p_in = _mean_or_zero(p_el_arr)
    mean_p_in_pos = _mean_or_zero(np.clip(p_el_arr, 0.0, None))

    denom = max(float(np.max(np.abs(ref_arr))) if ref_arr.size else 0.0, 1e-6)
    overshoot = float(np.max(omega_arr - ref_arr) / denom) if omega_arr.size else 0.0

    return {
        "mean_abs_speed_error": mean_abs_err,
        "iae_speed_error": iae,
        "ise_speed_error": ise,
        "mean_i_rms": mean_i_rms,
        "rms_i_rms": rms_i_rms,
        "mean_p_in": mean_p_in,
        "mean_p_in_pos": mean_p_in_pos,
        "overshoot_norm": overshoot,
        "dt_mean": float(dt),
    }


def score_from_metrics(
    metrics: Dict[str, float],
    fault_reason: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    fault_penalty: float = DEFAULT_FAULT_PENALTY,
) -> float:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    w_speed = float(weights.get("w_speed", DEFAULT_SCORE_WEIGHTS["w_speed"]))
    w_current = float(weights.get("w_current", DEFAULT_SCORE_WEIGHTS["w_current"]))
    w_power = float(weights.get("w_power", DEFAULT_SCORE_WEIGHTS["w_power"]))

    score = (
        w_speed * float(metrics.get("mean_abs_speed_error", 0.0))
        + w_current * float(metrics.get("mean_i_rms", 0.0))
        + w_power * float(metrics.get("mean_p_in_pos", 0.0))
    )
    if fault_reason:
        score += float(fault_penalty)
    return float(score)


__all__ = ["compute_metrics", "score_from_metrics", "DEFAULT_SCORE_WEIGHTS", "DEFAULT_FAULT_PENALTY"]
