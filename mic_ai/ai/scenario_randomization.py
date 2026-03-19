from __future__ import annotations

from typing import Callable

import numpy as np


def _sample_peak(func: Callable[[float], float], *, t_end: float, samples: int = 128) -> float:
    t_end = float(t_end)
    if not np.isfinite(t_end) or t_end <= 0.0:
        times = np.array([0.0], dtype=np.float64)
    else:
        times = np.linspace(0.0, t_end, max(int(samples), 2), dtype=np.float64)
    values = []
    for t in times:
        try:
            values.append(float(func(float(t))))
        except Exception:
            values.append(0.0)
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _scale_func(func: Callable[[float], float], scale: float) -> Callable[[float], float]:
    scale = float(scale)

    def _wrapped(t: float) -> float:
        try:
            value = float(func(float(t)))
        except Exception:
            value = 0.0
        return float(np.nan_to_num(scale * value, nan=0.0, posinf=0.0, neginf=0.0))

    return _wrapped


def _sample_target_peak(rng: np.random.Generator, rng_bounds: tuple[float, float] | None, base_peak: float) -> tuple[float, float]:
    base_peak = float(max(base_peak, 0.0))
    if rng_bounds is None or base_peak <= 1e-12:
        return 1.0, base_peak
    lo = float(min(rng_bounds))
    hi = float(max(rng_bounds))
    if hi <= 0.0:
        return 1.0, base_peak
    lo = max(lo, 0.0)
    target_peak = float(rng.uniform(lo, hi))
    if target_peak <= 0.0:
        return 1.0, base_peak
    return target_peak / max(base_peak, 1e-12), target_peak


def wrap_scenario_with_ranges(
    omega_ref_func: Callable[[float], float],
    load_torque_func: Callable[[float], float],
    *,
    t_end: float,
    rng: np.random.Generator,
    omega_ref_range: tuple[float, float] | None = None,
    load_torque_range: tuple[float, float] | None = None,
) -> tuple[Callable[[float], float], Callable[[float], float], dict[str, float]]:
    omega_base_peak = _sample_peak(omega_ref_func, t_end=t_end)
    load_base_peak = _sample_peak(load_torque_func, t_end=t_end)
    omega_scale, omega_peak = _sample_target_peak(rng, omega_ref_range, omega_base_peak)
    load_scale, load_peak = _sample_target_peak(rng, load_torque_range, load_base_peak)

    wrapped_omega = _scale_func(omega_ref_func, omega_scale)
    wrapped_load = _scale_func(load_torque_func, load_scale)
    meta = {
        "omega_base_peak": float(omega_base_peak),
        "load_base_peak": float(load_base_peak),
        "omega_scale": float(omega_scale),
        "load_scale": float(load_scale),
        "omega_peak": float(omega_peak if omega_ref_range is not None and omega_base_peak > 1e-12 else omega_base_peak),
        "load_peak": float(load_peak if load_torque_range is not None and load_base_peak > 1e-12 else load_base_peak),
    }
    return wrapped_omega, wrapped_load, meta
