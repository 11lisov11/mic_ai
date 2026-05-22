from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

from models.transformations import abc_to_alpha_beta, alpha_beta_to_abc


VectorBits = Tuple[int, int, int]


@dataclass(frozen=True)
class TwoLevelInverterParams:
    Vdc: float
    f_pwm: float = 10_000.0
    dead_time_s: float = 1.0e-6
    min_pulse_s: float = 2.0e-6
    r_on_ohm: float = 0.05
    v_drop_v: float = 0.8
    e_sw_j_per_a: float = 2.0e-6
    thermal_rth_k_per_w: float = 1.5
    thermal_cth_j_per_k: float = 10.0
    ambient_c: float = 25.0

    @property
    def t_pwm_s(self) -> float:
        return 1.0 / max(float(self.f_pwm), 1e-12)


@dataclass(frozen=True)
class InverterLossEstimate:
    conduction_w: float
    switching_w: float
    total_w: float
    switch_events: int
    common_mode_v: float


def validate_vector_id(vector_id: int) -> int:
    if not isinstance(vector_id, int):
        raise ValueError(f"vector_id must be int, got {vector_id!r}")
    if vector_id < 0 or vector_id > 7:
        raise ValueError(f"vector_id must be in 0..7, got {vector_id!r}")
    return vector_id


def vector_bits(vector_id: int) -> VectorBits:
    vector_id = validate_vector_id(vector_id)
    return ((vector_id >> 2) & 1, (vector_id >> 1) & 1, vector_id & 1)


def vector_id_from_bits(bits: Iterable[int]) -> int:
    a, b, c = tuple(int(x) for x in bits)
    for value in (a, b, c):
        if value not in (0, 1):
            raise ValueError(f"vector bit must be 0 or 1, got {value!r}")
    return (a << 2) | (b << 1) | c


def phase_voltages(vector_id: int, vdc: float) -> Tuple[float, float, float]:
    """Return motor phase voltages after common-mode removal."""

    sa, sb, sc = vector_bits(vector_id)
    mean_s = (sa + sb + sc) / 3.0
    return (
        (sa - mean_s) * float(vdc),
        (sb - mean_s) * float(vdc),
        (sc - mean_s) * float(vdc),
    )


def alpha_beta_voltage(
    vector_id: int,
    params: TwoLevelInverterParams,
    *,
    i_alpha_beta: Tuple[float, float] | None = None,
) -> Tuple[float, float]:
    v_a, v_b, v_c = phase_voltages(vector_id, params.Vdc)
    if i_alpha_beta is not None:
        i_a, i_b, i_c = alpha_beta_to_abc(float(i_alpha_beta[0]), float(i_alpha_beta[1]))
        dt_drop = abs(float(params.Vdc)) * max(float(params.dead_time_s), 0.0) / max(params.t_pwm_s, 1e-12)

        def sign(value: float) -> float:
            return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)

        v_a -= (dt_drop + float(params.v_drop_v)) * sign(i_a) + float(params.r_on_ohm) * i_a
        v_b -= (dt_drop + float(params.v_drop_v)) * sign(i_b) + float(params.r_on_ohm) * i_b
        v_c -= (dt_drop + float(params.v_drop_v)) * sign(i_c) + float(params.r_on_ohm) * i_c
    return abc_to_alpha_beta(v_a, v_b, v_c)


def common_mode_voltage(vector_id: int, vdc: float) -> float:
    sa, sb, sc = vector_bits(vector_id)
    return ((sa - 0.5) + (sb - 0.5) + (sc - 0.5)) * float(vdc) / 3.0


def switch_events(prev_vector_id: int, next_vector_id: int) -> int:
    prev = vector_bits(prev_vector_id)
    nxt = vector_bits(next_vector_id)
    return sum(1 for a, b in zip(prev, nxt) if a != b)


def estimate_inverter_losses(
    *,
    prev_vector_id: int,
    next_vector_id: int,
    params: TwoLevelInverterParams,
    i_alpha_beta: Tuple[float, float],
) -> InverterLossEstimate:
    i_a, i_b, i_c = alpha_beta_to_abc(float(i_alpha_beta[0]), float(i_alpha_beta[1]))
    i_rms_sq_sum = i_a * i_a + i_b * i_b + i_c * i_c
    conduction_w = max(float(params.r_on_ohm), 0.0) * i_rms_sq_sum
    events = switch_events(prev_vector_id, next_vector_id)
    i_abs = math.sqrt(max(i_rms_sq_sum / 3.0, 0.0))
    switching_w = events * max(float(params.e_sw_j_per_a), 0.0) * i_abs * max(float(params.f_pwm), 0.0)
    cmv = common_mode_voltage(next_vector_id, params.Vdc)
    return InverterLossEstimate(
        conduction_w=float(conduction_w),
        switching_w=float(switching_w),
        total_w=float(conduction_w + switching_w),
        switch_events=int(events),
        common_mode_v=float(cmv),
    )


__all__ = [
    "InverterLossEstimate",
    "TwoLevelInverterParams",
    "VectorBits",
    "alpha_beta_voltage",
    "common_mode_voltage",
    "estimate_inverter_losses",
    "phase_voltages",
    "switch_events",
    "validate_vector_id",
    "vector_bits",
    "vector_id_from_bits",
]
