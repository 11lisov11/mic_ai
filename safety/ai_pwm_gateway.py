from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntFlag
import math
from typing import Iterable, List, Sequence

from models.two_level_inverter import switch_events, validate_vector_id, vector_bits


class FaultFlag(IntFlag):
    NONE = 0
    OC_FAULT = 1 << 0
    DESAT_FAULT = 1 << 1
    UVLO_FAULT = 1 << 2
    OVERTEMP_FAULT = 1 << 3
    OVERVOLTAGE_FAULT = 1 << 4
    UNDERVOLTAGE_FAULT = 1 << 5
    WATCHDOG_FAULT = 1 << 6
    INVALID_VECTOR_FAULT = 1 << 7
    DEADTIME_FAULT = 1 << 8
    MIN_PULSE_FAULT = 1 << 9
    AI_CONFIDENCE_FAULT = 1 << 10
    CURRENT_SOFT_FAULT = 1 << 11
    SWITCHING_BUDGET_FAULT = 1 << 12
    OOD_FAULT = 1 << 13
    NONFINITE_FAULT = 1 << 14


CRITICAL_FAULTS = (
    FaultFlag.OC_FAULT
    | FaultFlag.DESAT_FAULT
    | FaultFlag.UVLO_FAULT
    | FaultFlag.OVERTEMP_FAULT
    | FaultFlag.OVERVOLTAGE_FAULT
    | FaultFlag.UNDERVOLTAGE_FAULT
    | FaultFlag.WATCHDOG_FAULT
    | FaultFlag.INVALID_VECTOR_FAULT
    | FaultFlag.DEADTIME_FAULT
    | FaultFlag.NONFINITE_FAULT
)


@dataclass(frozen=True)
class GateOutput:
    AH: bool = False
    AL: bool = False
    BH: bool = False
    BL: bool = False
    CH: bool = False
    CL: bool = False

    @property
    def shoot_through(self) -> bool:
        return (self.AH and self.AL) or (self.BH and self.BL) or (self.CH and self.CL)


@dataclass(frozen=True)
class AIPwmRequest:
    vector_id: int
    dwell_s: float
    confidence: float
    predicted_i_abs: float
    measured_i_abs: float
    vdc: float
    tj_c: float
    predicted_risk: float = 0.0
    watchdog_ok: bool = True


@dataclass(frozen=True)
class GatewayLimits:
    t_pwm_s: float = 100.0e-6
    dead_time_s: float = 1.0e-6
    min_pulse_s: float = 2.0e-6
    i_soft_a: float = 8.0
    i_trip_a: float = 12.0
    vdc_min_v: float = 40.0
    vdc_max_v: float = 900.0
    tj_trip_c: float = 125.0
    confidence_min: float = 0.4
    risk_max: float = 1.0
    max_switch_events_per_window: int = 12
    switch_window_steps: int = 8


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    pwm_enabled: bool
    vector_id: int
    requested_vector_id: int
    gates: GateOutput
    fault_flags: FaultFlag
    fault_latched: bool
    fallback_reason: str = ""


def gates_from_vector(vector_id: int, *, pwm_enabled: bool = True) -> GateOutput:
    if not pwm_enabled:
        return GateOutput()
    sa, sb, sc = vector_bits(vector_id)
    return GateOutput(
        AH=bool(sa),
        AL=not bool(sa),
        BH=bool(sb),
        BL=not bool(sb),
        CH=bool(sc),
        CL=not bool(sc),
    )


def transition_waveform(prev_vector_id: int, next_vector_id: int, dead_time_ticks: int = 1) -> List[GateOutput]:
    prev_bits = vector_bits(prev_vector_id)
    next_bits = vector_bits(next_vector_id)
    dead_time_ticks = max(int(dead_time_ticks), 0)
    wave: List[GateOutput] = [gates_from_vector(prev_vector_id)]
    if prev_bits == next_bits:
        return wave

    for _ in range(dead_time_ticks):
        leg_values: list[tuple[bool, bool]] = []
        for old, new in zip(prev_bits, next_bits):
            if old == new:
                leg_values.append((bool(old), not bool(old)))
            else:
                leg_values.append((False, False))
        wave.append(
            GateOutput(
                AH=leg_values[0][0],
                AL=leg_values[0][1],
                BH=leg_values[1][0],
                BL=leg_values[1][1],
                CH=leg_values[2][0],
                CL=leg_values[2][1],
            )
        )
    wave.append(gates_from_vector(next_vector_id))
    return wave


def has_shoot_through(waveform: Iterable[GateOutput]) -> bool:
    return any(item.shoot_through for item in waveform)


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _fault_names(flags: FaultFlag) -> str:
    if flags == FaultFlag.NONE:
        return ""
    return ",".join(flag.name for flag in FaultFlag if flag is not FaultFlag.NONE and flag in flags)


class AIPwmSafetyGateway:
    """Host-level protected AI-PWM gatekeeper with no raw low-switch access."""

    def __init__(self, limits: GatewayLimits | None = None, initial_vector_id: int = 0) -> None:
        self.limits = limits if limits is not None else GatewayLimits()
        self.current_vector_id = validate_vector_id(int(initial_vector_id))
        self.fault_latched = False
        self._switch_window: deque[int] = deque(maxlen=max(int(self.limits.switch_window_steps), 1))

    def reset(self, initial_vector_id: int = 0) -> None:
        self.current_vector_id = validate_vector_id(int(initial_vector_id))
        self.fault_latched = False
        self._switch_window.clear()

    def clear_fault_latch(self) -> None:
        self.fault_latched = False

    def _validate_request(self, req: AIPwmRequest) -> FaultFlag:
        flags = FaultFlag.NONE
        if not _finite(req.dwell_s) or not _finite(req.confidence) or not _finite(req.predicted_i_abs):
            flags |= FaultFlag.NONFINITE_FAULT
        if not _finite(req.measured_i_abs) or not _finite(req.vdc) or not _finite(req.tj_c):
            flags |= FaultFlag.NONFINITE_FAULT
        if not _finite(req.predicted_risk):
            flags |= FaultFlag.NONFINITE_FAULT

        try:
            validate_vector_id(req.vector_id)
        except Exception:
            flags |= FaultFlag.INVALID_VECTOR_FAULT

        if not bool(req.watchdog_ok):
            flags |= FaultFlag.WATCHDOG_FAULT
        if _finite(req.dwell_s) and float(req.dwell_s) < float(self.limits.min_pulse_s):
            flags |= FaultFlag.MIN_PULSE_FAULT
        if _finite(req.predicted_i_abs) and abs(float(req.predicted_i_abs)) >= float(self.limits.i_soft_a):
            flags |= FaultFlag.CURRENT_SOFT_FAULT
        if _finite(req.measured_i_abs) and abs(float(req.measured_i_abs)) >= float(self.limits.i_trip_a):
            flags |= FaultFlag.OC_FAULT
        if _finite(req.vdc) and float(req.vdc) <= float(self.limits.vdc_min_v):
            flags |= FaultFlag.UNDERVOLTAGE_FAULT
        if _finite(req.vdc) and float(req.vdc) >= float(self.limits.vdc_max_v):
            flags |= FaultFlag.OVERVOLTAGE_FAULT
        if _finite(req.tj_c) and float(req.tj_c) >= float(self.limits.tj_trip_c):
            flags |= FaultFlag.OVERTEMP_FAULT
        if _finite(req.confidence) and float(req.confidence) < float(self.limits.confidence_min):
            flags |= FaultFlag.AI_CONFIDENCE_FAULT
        if _finite(req.predicted_risk) and float(req.predicted_risk) > float(self.limits.risk_max):
            flags |= FaultFlag.OOD_FAULT

        if FaultFlag.INVALID_VECTOR_FAULT not in flags:
            events = switch_events(self.current_vector_id, int(req.vector_id))
            projected = sum(self._switch_window) + events
            if projected > int(self.limits.max_switch_events_per_window):
                flags |= FaultFlag.SWITCHING_BUDGET_FAULT

        return flags

    def evaluate(self, req: AIPwmRequest) -> GateDecision:
        requested = int(req.vector_id) if isinstance(req.vector_id, int) else 0
        flags = self._validate_request(req)
        if flags & CRITICAL_FAULTS:
            self.fault_latched = True
        if self.fault_latched:
            flags |= FaultFlag.WATCHDOG_FAULT if not bool(req.watchdog_ok) else FaultFlag.NONE
            return GateDecision(
                accepted=False,
                pwm_enabled=False,
                vector_id=self.current_vector_id,
                requested_vector_id=requested,
                gates=GateOutput(),
                fault_flags=flags,
                fault_latched=True,
                fallback_reason=_fault_names(flags) or "fault_latched",
            )

        if flags == FaultFlag.NONE:
            events = switch_events(self.current_vector_id, requested)
            self._switch_window.append(events)
            self.current_vector_id = requested
            return GateDecision(
                accepted=True,
                pwm_enabled=True,
                vector_id=requested,
                requested_vector_id=requested,
                gates=gates_from_vector(requested),
                fault_flags=FaultFlag.NONE,
                fault_latched=False,
            )

        safe_vector = self.current_vector_id
        self._switch_window.append(0)
        return GateDecision(
            accepted=False,
            pwm_enabled=True,
            vector_id=safe_vector,
            requested_vector_id=requested,
            gates=gates_from_vector(safe_vector),
            fault_flags=flags,
            fault_latched=False,
            fallback_reason=_fault_names(flags),
        )

    def valid_action_mask(self, template: AIPwmRequest) -> list[bool]:
        out: list[bool] = []
        for vector_id in range(8):
            req = AIPwmRequest(
                vector_id=vector_id,
                dwell_s=template.dwell_s,
                confidence=template.confidence,
                predicted_i_abs=template.predicted_i_abs,
                measured_i_abs=template.measured_i_abs,
                vdc=template.vdc,
                tj_c=template.tj_c,
                predicted_risk=template.predicted_risk,
                watchdog_ok=template.watchdog_ok,
            )
            out.append(self._validate_request(req) == FaultFlag.NONE and not self.fault_latched)
        return out

    def evaluate_sequence(self, requests: Sequence[AIPwmRequest]) -> list[GateDecision]:
        return [self.evaluate(req) for req in requests]


__all__ = [
    "AIPwmRequest",
    "AIPwmSafetyGateway",
    "CRITICAL_FAULTS",
    "FaultFlag",
    "GateDecision",
    "GateOutput",
    "GatewayLimits",
    "gates_from_vector",
    "has_shoot_through",
    "transition_waveform",
]
