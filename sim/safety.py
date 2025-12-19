from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SafetyLimits:
    """Safety limits for the bench."""

    i_inst_max: float | None = None
    i_rms_max: float | None = None
    omega_max: float | None = None
    vdc_max: float | None = None


class SafetySupervisor:
    """Checks safety limits and returns a stop reason on violation."""

    def __init__(self, limits: SafetyLimits) -> None:
        self._limits = limits
        self._validate_limits()

    def reset(self) -> None:
        """Reset internal state (reserved for future RMS tracking)."""
        return None

    @property
    def limits(self) -> SafetyLimits:
        """Expose configured limits for logging."""
        return self._limits

    def check(self, state: Mapping[str, float], signals: Mapping[str, object]) -> tuple[bool, str]:
        """Validate safety constraints.

        Args:
            state: State dictionary (expects omega when omega_max is set).
            signals: Signals dictionary (expects currents and vdc when limits are set).
        """
        omega = _get_scalar(state, signals, ("omega", "omega_m", "w_mech"))
        if self._limits.omega_max is not None:
            if omega is None:
                return False, "omega missing for omega_max check"
            if abs(omega) > self._limits.omega_max:
                return False, f"omega_max exceeded: omega={omega:.6f} limit={self._limits.omega_max:.6f}"

        vdc = _get_scalar(state, signals, ("vdc", "udc", "u_dc"))
        if self._limits.vdc_max is not None:
            if vdc is None:
                return False, "vdc missing for vdc_max check"
            if vdc > self._limits.vdc_max:
                return False, f"vdc_max exceeded: vdc={vdc:.6f} limit={self._limits.vdc_max:.6f}"

        i_inst = _get_instant_current(signals)
        if self._limits.i_inst_max is not None:
            if i_inst is None:
                return False, "i_inst missing for i_inst_max check"
            if i_inst > self._limits.i_inst_max:
                return False, (
                    "i_inst_max exceeded: "
                    f"i_inst={i_inst:.6f} limit={self._limits.i_inst_max:.6f}"
                )

        i_rms = _get_rms_current(signals)
        if self._limits.i_rms_max is not None:
            if i_rms is None:
                return False, "i_rms missing for i_rms_max check"
            if i_rms > self._limits.i_rms_max:
                return False, f"i_rms_max exceeded: i_rms={i_rms:.6f} limit={self._limits.i_rms_max:.6f}"

        return True, ""

    def _validate_limits(self) -> None:
        for name, value in (
            ("i_inst_max", self._limits.i_inst_max),
            ("i_rms_max", self._limits.i_rms_max),
            ("omega_max", self._limits.omega_max),
            ("vdc_max", self._limits.vdc_max),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when set")


def _get_scalar(
    state: Mapping[str, float],
    signals: Mapping[str, object],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        if key in state:
            return float(state[key])
        if key in signals:
            return float(signals[key])  # type: ignore[call-arg]
    return None


def _get_instant_current(signals: Mapping[str, object]) -> float | None:
    if "i_inst" in signals:
        return abs(float(signals["i_inst"]))  # type: ignore[call-arg]
    if "i_mag" in signals:
        return abs(float(signals["i_mag"]))  # type: ignore[call-arg]
    if "i_dq" in signals:
        dq = signals["i_dq"]
        if isinstance(dq, Sequence) and len(dq) >= 2:
            id_val = float(dq[0])
            iq_val = float(dq[1])
            return (id_val * id_val + iq_val * iq_val) ** 0.5
    if "i_abc" in signals:
        abc = signals["i_abc"]
        if isinstance(abc, Sequence) and len(abc) >= 3:
            return max(abs(float(abc[0])), abs(float(abc[1])), abs(float(abc[2])))
    return None


def _get_rms_current(signals: Mapping[str, object]) -> float | None:
    if "i_rms" in signals:
        return abs(float(signals["i_rms"]))  # type: ignore[call-arg]
    if "i_abc_rms" in signals:
        abc = signals["i_abc_rms"]
        if isinstance(abc, Sequence) and len(abc) >= 3:
            return max(abs(float(abc[0])), abs(float(abc[1])), abs(float(abc[2])))
    return None


__all__ = ["SafetyLimits", "SafetySupervisor"]
