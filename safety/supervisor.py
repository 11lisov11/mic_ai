from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Optional, Tuple


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be float-like, got {value!r}") from exc


def _normalize_key(key: Any) -> Optional[str]:
    raw = str(key).strip().lower()
    raw = raw.replace(" ", "").replace("-", "_")
    raw = raw.replace("/", "")
    mapping = {
        "i_max": "i_max",
        "imax": "i_max",
        "current_max": "i_max",
        "v_max": "v_max",
        "vmax": "v_max",
        "u_max": "v_max",
        "umax": "v_max",
        "voltage_max": "v_max",
        "dv_dt": "dv_dt",
        "dvdt": "dv_dt",
        "omega_max": "omega_max",
        "w_max": "omega_max",
        "speed_max": "omega_max",
        "t_max": "temp_max",
        "temp_max": "temp_max",
        "temperature_max": "temp_max",
    }
    return mapping.get(raw)


@dataclass(frozen=True)
class SafetyLimits:
    i_max: Optional[float] = None
    v_max: Optional[float] = None
    dv_dt: Optional[float] = None
    omega_max: Optional[float] = None
    temp_max: Optional[float] = None


def _merge_limits(base: SafetyLimits, updates: Dict[str, Any]) -> SafetyLimits:
    limits = base
    for key, value in updates.items():
        norm = _normalize_key(key)
        if norm is None:
            continue
        if value is None:
            continue
        limits = replace(limits, **{norm: _as_float(value, norm)})
    return limits


class SafetySupervisor:
    """Clip commands and detect limit violations."""

    def __init__(self, dt: float, limits: SafetyLimits | None = None) -> None:
        self._dt = float(dt)
        self._limits = limits if limits is not None else SafetyLimits()
        self._last_action = (0.0, 0.0)
        self._last_fault: Optional[str] = None

    @property
    def limits(self) -> SafetyLimits:
        return self._limits

    def configure(self, limits: SafetyLimits | Dict[str, Any]) -> None:
        if isinstance(limits, SafetyLimits):
            self._limits = limits
        elif isinstance(limits, dict):
            self._limits = _merge_limits(self._limits, limits)
        else:
            raise TypeError("limits must be SafetyLimits or dict")

    def reset(self) -> None:
        self._last_action = (0.0, 0.0)
        self._last_fault = None

    def last_fault(self) -> Optional[str]:
        return self._last_fault

    def sanitize_action(
        self,
        action: Tuple[float, float],
        obs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        vd, vq = float(action[0]), float(action[1])
        if not _is_finite(vd) or not _is_finite(vq):
            self._last_fault = "action_non_finite"
            return 0.0, 0.0

        if self._limits.v_max is not None and self._limits.v_max > 0.0:
            mag = math.hypot(vd, vq)
            if mag > self._limits.v_max and mag > 0.0:
                scale = self._limits.v_max / mag
                vd *= scale
                vq *= scale

        if self._limits.dv_dt is not None and self._limits.dv_dt > 0.0 and self._dt > 0.0:
            dv = math.hypot(vd - self._last_action[0], vq - self._last_action[1])
            max_delta = self._limits.dv_dt * self._dt
            if dv > max_delta and dv > 0.0:
                scale = max_delta / dv
                vd = self._last_action[0] + (vd - self._last_action[0]) * scale
                vq = self._last_action[1] + (vq - self._last_action[1]) * scale

        self._last_action = (vd, vq)
        return vd, vq

    def check_abort(self, obs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if obs is None:
            return False, None

        for key in ("omega", "id", "iq", "ia", "ib", "ic", "u_dc", "theta_e", "torque", "temp"):
            if key in obs and obs[key] is not None and not _is_finite(obs[key]):
                reason = f"obs_non_finite:{key}"
                self._last_fault = reason
                return True, reason

        i_rms = None
        if all(k in obs for k in ("ia", "ib", "ic")):
            ia, ib, ic = float(obs["ia"]), float(obs["ib"]), float(obs["ic"])
            i_rms = math.sqrt((ia * ia + ib * ib + ic * ic) / 3.0)
        if i_rms is None and "id" in obs and "iq" in obs:
            id_val, iq_val = float(obs["id"]), float(obs["iq"])
            i_rms = math.hypot(id_val, iq_val)

        if i_rms is not None and self._limits.i_max is not None and self._limits.i_max > 0.0:
            if i_rms > self._limits.i_max:
                reason = "over_current"
                self._last_fault = reason
                return True, reason

        if self._limits.omega_max is not None and self._limits.omega_max > 0.0:
            omega = float(obs.get("omega", 0.0))
            if abs(omega) > self._limits.omega_max:
                reason = "over_speed"
                self._last_fault = reason
                return True, reason

        if self._limits.temp_max is not None and self._limits.temp_max > 0.0:
            if "temp" in obs and obs["temp"] is not None:
                temp = float(obs["temp"])
                if temp > self._limits.temp_max:
                    reason = "over_temp"
                    self._last_fault = reason
                    return True, reason

        return False, None


__all__ = ["SafetySupervisor", "SafetyLimits"]
