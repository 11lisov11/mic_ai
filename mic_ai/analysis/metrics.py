from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np


def _as_abc_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 1:
        if arr.size == 3:
            return arr.reshape(1, 3)
        if arr.size % 3 == 0:
            return arr.reshape(-1, 3)
        raise ValueError(f"Expected abc vector size 3 or multiple of 3, got {arr.size}")
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(f"Expected shape (N,3) or (3,), got {arr.shape}")


def calc_i_rms(i_abc: Iterable[float]) -> float:
    values = _as_abc_array(i_abc)
    if values.size == 0:
        return 0.0
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.clip(safe, -1e100, 1e100)
    # Phase RMS, averaged over all three phases.
    i_rms_phase = np.sqrt(np.mean(safe * safe, axis=0))
    return float(np.mean(i_rms_phase))


def calc_v_rms(v_abc: Iterable[float]) -> float:
    values = _as_abc_array(v_abc)
    if values.size == 0:
        return 0.0
    safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.clip(safe, -1e100, 1e100)
    # Phase RMS, averaged over all three phases.
    v_rms_phase = np.sqrt(np.mean(safe * safe, axis=0))
    return float(np.mean(v_rms_phase))


def calc_p_el(v_abc: Iterable[float], i_abc: Iterable[float]) -> float:
    v_vals = np.asarray(v_abc, dtype=float)
    i_vals = np.asarray(i_abc, dtype=float)
    if v_vals.size == 0 or i_vals.size == 0:
        return 0.0
    v_safe = np.nan_to_num(v_vals, nan=0.0, posinf=0.0, neginf=0.0)
    i_safe = np.nan_to_num(i_vals, nan=0.0, posinf=0.0, neginf=0.0)
    v_safe = np.clip(v_safe, -1e100, 1e100)
    i_safe = np.clip(i_safe, -1e100, 1e100)
    return float(np.sum(v_safe * i_safe))


def calc_p_mech(omega: float, torque: float) -> float:
    max_abs = 1e6
    max_power = 1e12
    omega_safe = float(np.nan_to_num(float(omega), nan=0.0, posinf=max_abs, neginf=-max_abs))
    torque_safe = float(np.nan_to_num(float(torque), nan=0.0, posinf=max_abs, neginf=-max_abs))
    omega_safe = float(np.clip(omega_safe, -max_abs, max_abs))
    torque_safe = float(np.clip(torque_safe, -max_abs, max_abs))
    power = omega_safe * torque_safe
    power = float(np.nan_to_num(power, nan=0.0, posinf=max_power, neginf=-max_power))
    return float(np.clip(power, -max_power, max_power))


def calc_eta(p_mech: float, p_in: float, eps: float = 1e-9) -> float:
    p2 = float(p_mech)
    p1 = float(p_in)
    if not math.isfinite(p2) or not math.isfinite(p1) or p1 <= eps:
        return 0.0
    return float(np.clip(p2 / max(p1, eps), 0.0, 1.0))


def calc_cos_phi(
    v_abc: Iterable[float],
    i_abc: Iterable[float],
    window_slice: Optional[slice] = None,
    eps: float = 1e-9,
) -> Tuple[float, dict]:
    """
    Estimate power factor from instantaneous 3-phase voltages/currents.

    Historical pitfall fixed here:
    older code assumed one apparent-power formula for all datasets and could
    mix phase and line-line voltage conventions, which distorted cosphi shape.
    This function evaluates both hypotheses and reports diagnostics.

    Returns:
        (cos_phi, diagnostics)
    """
    v = _as_abc_array(v_abc)
    i = _as_abc_array(i_abc)
    if v.size == 0 or i.size == 0:
        return 0.0, {"method": "none", "warning": "empty_inputs"}
    n = int(min(v.shape[0], i.shape[0]))
    v = v[:n]
    i = i[:n]
    if window_slice is not None:
        v = v[window_slice]
        i = i[window_slice]
    if v.size == 0 or i.size == 0:
        return 0.0, {"method": "none", "warning": "empty_window"}

    p_inst = np.sum(v * i, axis=1)
    p_mean = float(np.mean(p_inst))
    i_rms_phase = float(calc_i_rms(i))
    v_rms_phase = float(calc_v_rms(v))
    s_phase = float(3.0 * v_rms_phase * i_rms_phase)
    cos_phase_raw = float(p_mean / max(s_phase, eps))
    cos_phase = float(np.clip(cos_phase_raw, 0.0, 1.0))

    # Alternative hypothesis: provided voltages are line-line channels (Vab/Vbc/Vca).
    v_ll_rms_given = float(np.mean(np.sqrt(np.mean(v * v, axis=0))))
    s_line = float(math.sqrt(3.0) * v_ll_rms_given * i_rms_phase)
    cos_line_raw = float(p_mean / max(s_line, eps))
    cos_line = float(np.clip(cos_line_raw, 0.0, 1.0))

    phase_ok = bool(np.isfinite(cos_phase_raw) and (-0.2 <= cos_phase_raw <= 1.2))
    line_ok = bool(np.isfinite(cos_line_raw) and (-0.2 <= cos_line_raw <= 1.2))

    method = "phase"
    warning = ""
    if phase_ok and not line_ok:
        method = "phase"
    elif line_ok and not phase_ok:
        method = "line"
    elif not phase_ok and not line_ok:
        # Both are suspicious; keep phase as default fallback.
        method = "phase"
        warning = "both_phase_and_line_out_of_range"
    else:
        # Both plausible; choose the one requiring less clipping.
        clip_pen_phase = abs(cos_phase_raw - cos_phase)
        clip_pen_line = abs(cos_line_raw - cos_line)
        if clip_pen_line + 1e-12 < clip_pen_phase:
            method = "line"
        else:
            method = "phase"
        if abs(cos_phase_raw - cos_line_raw) > 0.05:
            warning = "phase_line_disagreement"

    cos_phi = cos_phase if method == "phase" else cos_line
    if not math.isfinite(cos_phi):
        cos_phi = 0.0
        warning = (warning + "|non_finite_cosphi").strip("|")
        diag_warning = warning
    else:
        diag_warning = warning
    diag = {
        "method": method,
        "warning": diag_warning,
        "p_mean": p_mean,
        "v_rms_phase": v_rms_phase,
        "i_rms_phase": i_rms_phase,
        "v_ll_rms_given": v_ll_rms_given,
        "s_phase": s_phase,
        "s_line": s_line,
        "cos_phase_raw": cos_phase_raw,
        "cos_line_raw": cos_line_raw,
        "cos_phase": cos_phase,
        "cos_line": cos_line,
    }
    return float(np.clip(cos_phi, 0.0, 1.0)), diag
