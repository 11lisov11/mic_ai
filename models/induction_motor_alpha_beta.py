from __future__ import annotations

from dataclasses import dataclass, replace
import math
from random import Random
from typing import Mapping

from config.env import MotorParams


@dataclass(frozen=True)
class AlphaBetaMotorParams:
    Rs: float
    Rr: float
    Lls: float
    Llr: float
    Lm: float
    J: float
    B: float
    p: int
    i_limit: float = 10.0
    psi_sat: float = 0.0
    sat_exp: float = 2.0
    lm_min_scale: float = 0.25
    rs_temp_coeff: float = 0.0039
    rr_temp_coeff: float = 0.0039
    temp_ref_c: float = 20.0

    @classmethod
    def from_motor_params(cls, params: MotorParams) -> "AlphaBetaMotorParams":
        return cls(
            Rs=float(params.Rs),
            Rr=float(params.Rr),
            Lls=float(params.Ls_sigma),
            Llr=float(params.Lr_sigma),
            Lm=float(params.Lm),
            J=float(params.J),
            B=float(params.B),
            p=int(params.p),
            i_limit=float(getattr(params, "I_n", 10.0) or 10.0) * 2.5,
            psi_sat=float(getattr(params, "psi_sat", 0.0) or 0.0),
            sat_exp=float(getattr(params, "sat_exp", 2.0) or 2.0),
            lm_min_scale=float(getattr(params, "lm_min_scale", 0.25) or 0.25),
        )


@dataclass(frozen=True)
class AlphaBetaMotorState:
    psi_s_alpha: float = 0.0
    psi_s_beta: float = 0.0
    psi_r_alpha: float = 0.0
    psi_r_beta: float = 0.0
    omega_m: float = 0.0
    theta_m: float = 0.0
    temp_s_c: float = 20.0
    temp_r_c: float = 20.0


@dataclass(frozen=True)
class AlphaBetaCurrents:
    i_s_alpha: float
    i_s_beta: float
    i_r_alpha: float
    i_r_beta: float

    @property
    def stator_abs(self) -> float:
        return math.hypot(self.i_s_alpha, self.i_s_beta)


@dataclass(frozen=True)
class AlphaBetaStep:
    state: AlphaBetaMotorState
    currents: AlphaBetaCurrents
    torque_nm: float
    p_mech_w: float


def _finite_or(value: float, fallback: float) -> float:
    try:
        value = float(value)
    except Exception:
        return fallback
    return value if math.isfinite(value) else fallback


def _effective_lm(params: AlphaBetaMotorParams, state: AlphaBetaMotorState) -> float:
    if params.psi_sat <= 0.0:
        return float(params.Lm)
    psi_abs = math.hypot(state.psi_s_alpha, state.psi_s_beta)
    if psi_abs <= 0.0:
        return float(params.Lm)
    scale = 1.0 / (1.0 + (psi_abs / params.psi_sat) ** max(params.sat_exp, 1e-9))
    scale = max(float(params.lm_min_scale), min(1.0, scale))
    return float(params.Lm) * scale


def _temp_scaled_r(base: float, coeff: float, temp_c: float, temp_ref_c: float) -> float:
    return float(base) * (1.0 + float(coeff) * (float(temp_c) - float(temp_ref_c)))


class AlphaBetaInductionMotorModel:
    """Stationary alpha-beta flux model for host-level research simulations."""

    def __init__(self, params: AlphaBetaMotorParams, state: AlphaBetaMotorState | None = None) -> None:
        self.params = params
        self.state = state if state is not None else AlphaBetaMotorState()

    def copy(self) -> "AlphaBetaInductionMotorModel":
        return AlphaBetaInductionMotorModel(self.params, self.state)

    def currents(
        self,
        state: AlphaBetaMotorState | None = None,
        params: AlphaBetaMotorParams | None = None,
    ) -> AlphaBetaCurrents:
        state = state if state is not None else self.state
        params = params if params is not None else self.params
        lm = _effective_lm(params, state)
        ls = float(params.Lls) + lm
        lr = float(params.Llr) + lm
        denom = ls * lr - lm * lm
        if abs(denom) < 1e-12:
            denom = math.copysign(1e-12, denom if denom != 0.0 else 1.0)

        i_s_alpha = (state.psi_s_alpha * lr - state.psi_r_alpha * lm) / denom
        i_s_beta = (state.psi_s_beta * lr - state.psi_r_beta * lm) / denom
        i_r_alpha = (state.psi_r_alpha * ls - state.psi_s_alpha * lm) / denom
        i_r_beta = (state.psi_r_beta * ls - state.psi_s_beta * lm) / denom
        return AlphaBetaCurrents(i_s_alpha, i_s_beta, i_r_alpha, i_r_beta)

    def torque_nm(
        self,
        state: AlphaBetaMotorState | None = None,
        currents: AlphaBetaCurrents | None = None,
        params: AlphaBetaMotorParams | None = None,
    ) -> float:
        state = state if state is not None else self.state
        params = params if params is not None else self.params
        currents = currents if currents is not None else self.currents(state, params)
        return 1.5 * float(params.p) * (
            state.psi_s_alpha * currents.i_s_beta - state.psi_s_beta * currents.i_s_alpha
        )

    def next_state(
        self,
        v_alpha: float,
        v_beta: float,
        load_torque_nm: float,
        dt: float,
        *,
        state: AlphaBetaMotorState | None = None,
        params: AlphaBetaMotorParams | None = None,
    ) -> AlphaBetaStep:
        state = state if state is not None else self.state
        params = params if params is not None else self.params
        v_alpha = _finite_or(v_alpha, 0.0)
        v_beta = _finite_or(v_beta, 0.0)
        load_torque_nm = _finite_or(load_torque_nm, 0.0)
        dt = max(_finite_or(dt, 0.0), 0.0)

        currents = self.currents(state, params)
        rs = _temp_scaled_r(params.Rs, params.rs_temp_coeff, state.temp_s_c, params.temp_ref_c)
        rr = _temp_scaled_r(params.Rr, params.rr_temp_coeff, state.temp_r_c, params.temp_ref_c)
        omega_r_e = float(params.p) * state.omega_m

        dpsi_s_alpha = v_alpha - rs * currents.i_s_alpha
        dpsi_s_beta = v_beta - rs * currents.i_s_beta
        dpsi_r_alpha = -rr * currents.i_r_alpha - omega_r_e * state.psi_r_beta
        dpsi_r_beta = -rr * currents.i_r_beta + omega_r_e * state.psi_r_alpha

        torque = self.torque_nm(state, currents, params)
        j = max(float(params.J), 1e-12)
        domega_m = (torque - load_torque_nm - float(params.B) * state.omega_m) / j

        next_state = replace(
            state,
            psi_s_alpha=state.psi_s_alpha + dt * dpsi_s_alpha,
            psi_s_beta=state.psi_s_beta + dt * dpsi_s_beta,
            psi_r_alpha=state.psi_r_alpha + dt * dpsi_r_alpha,
            psi_r_beta=state.psi_r_beta + dt * dpsi_r_beta,
            omega_m=state.omega_m + dt * domega_m,
            theta_m=state.theta_m + dt * state.omega_m,
        )
        next_currents = self.currents(next_state, params)
        next_torque = self.torque_nm(next_state, next_currents, params)
        return AlphaBetaStep(
            state=next_state,
            currents=next_currents,
            torque_nm=next_torque,
            p_mech_w=next_torque * next_state.omega_m,
        )

    def step(self, v_alpha: float, v_beta: float, load_torque_nm: float, dt: float) -> AlphaBetaStep:
        result = self.next_state(v_alpha, v_beta, load_torque_nm, dt)
        self.state = result.state
        return result


def randomized_motor_params(
    base: AlphaBetaMotorParams,
    rng: Random,
    *,
    rs_span: float = 0.5,
    rr_span: float = 0.5,
    lm_span: float = 0.2,
    j_span: float = 1.0,
    b_span: float = 1.0,
    extra: Mapping[str, float] | None = None,
) -> AlphaBetaMotorParams:
    """Domain randomization profile from the research TZ; spans are fractional."""

    def scale(span: float) -> float:
        span = max(float(span), 0.0)
        return 1.0 + rng.uniform(-span, span)

    overrides = dict(extra or {})
    return replace(
        base,
        Rs=max(1e-9, base.Rs * scale(float(overrides.get("rs_span", rs_span)))),
        Rr=max(1e-9, base.Rr * scale(float(overrides.get("rr_span", rr_span)))),
        Lm=max(1e-9, base.Lm * scale(float(overrides.get("lm_span", lm_span)))),
        J=max(1e-9, base.J * scale(float(overrides.get("j_span", j_span)))),
        B=max(0.0, base.B * scale(float(overrides.get("b_span", b_span)))),
    )


__all__ = [
    "AlphaBetaCurrents",
    "AlphaBetaInductionMotorModel",
    "AlphaBetaMotorParams",
    "AlphaBetaMotorState",
    "AlphaBetaStep",
    "randomized_motor_params",
]
