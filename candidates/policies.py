from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from config.env import EnvConfig, FocParams, MotorParams, ScalarVfParams
from control.scalar_vf import ScalarVfController
from control.vector_foc import FocController


@dataclass(frozen=True)
class PolicyAction:
    """Policy action in dq frame."""

    v_d: float
    v_q: float
    theta_e: float
    omega_syn: float


class VfPolicy:
    """Wrapper around scalar V/f controller with policy.act interface."""

    def __init__(self, params: ScalarVfParams, dt: float, p: int, vdc: float) -> None:
        self._controller = ScalarVfController(params, dt, p, vdc)

    def reset(self) -> None:
        self._controller.reset()

    def act(self, state: Mapping[str, object]) -> PolicyAction:
        v_d, v_q, theta_e, omega_syn, _ = self._controller.step(
            float(state["t"]),
            float(state["omega_ref"]),
            float(state["omega_m"]),
            _coerce_tuple(state["i_abc"]),
            float(state["torque_e"]),
            float(state["theta_mech"]),
        )
        return PolicyAction(v_d=v_d, v_q=v_q, theta_e=theta_e, omega_syn=omega_syn)


class FocPolicy:
    """Wrapper around FOC controller with policy.act interface."""

    def __init__(self, params: FocParams, env: EnvConfig, dt: float) -> None:
        self._controller = FocController(params, env.motor, dt)

    def reset(self) -> None:
        self._controller.reset()

    def act(self, state: Mapping[str, object]) -> PolicyAction:
        v_d, v_q, theta_e, omega_syn, _ = self._controller.step(
            float(state["t"]),
            float(state["omega_ref"]),
            float(state["omega_m"]),
            _coerce_tuple(state["i_abc"]),
            float(state["torque_e"]),
            float(state["theta_mech"]),
        )
        return PolicyAction(v_d=v_d, v_q=v_q, theta_e=theta_e, omega_syn=omega_syn)


class NnStubPolicy:
    """Deterministic stub policy (zero voltages)."""

    def __init__(self, v_d: float = 0.0, v_q: float = 0.0) -> None:
        self._v_d = v_d
        self._v_q = v_q

    def reset(self) -> None:
        return None

    def act(self, state: Mapping[str, object]) -> PolicyAction:
        return PolicyAction(v_d=self._v_d, v_q=self._v_q, theta_e=0.0, omega_syn=0.0)


class PIDSpeedPolicy:
    """Speed PI policy implemented via the FOC speed loop."""

    def __init__(self, params: FocParams, motor: MotorParams, dt: float) -> None:
        self._controller = FocController(params, motor, dt)

    def reset(self) -> None:
        self._controller.reset()

    def act(self, state: Mapping[str, object]) -> PolicyAction:
        v_d, v_q, theta_e, omega_syn, _ = self._controller.step(
            float(state["t"]),
            float(state["omega_ref"]),
            float(state["omega_m"]),
            _coerce_tuple(state["i_abc"]),
            float(state["torque_e"]),
            float(state["theta_mech"]),
        )
        return PolicyAction(v_d=v_d, v_q=v_q, theta_e=theta_e, omega_syn=omega_syn)


class MICRulePolicy:
    """Rule-based MIC policy with aging-aware gain scheduling."""

    def __init__(
        self,
        params: FocParams,
        motor: MotorParams,
        dt: float,
        vq_max: float | None,
        dvq_max: float | None = None,
        tau_action: float | None = None,
        k_age: float = 0.0,
    ) -> None:
        self._controller = FocController(params, motor, dt)
        self._base_kp = float(params.kp_speed)
        self._base_ki = float(params.ki_speed)
        self._vq_max = vq_max
        self._dvq_max = dvq_max
        self._tau_action = tau_action
        self._k_age = k_age
        self._dt = dt
        self._vq_prev = 0.0

    def reset(self) -> None:
        self._controller.reset()
        self._vq_prev = 0.0

    def act(self, state: Mapping[str, object]) -> PolicyAction:
        aging = float(state.get("aging", 0.0))
        gain_sched = max(0.0, 1.0 + self._k_age * aging)
        self._controller.pi_speed.kp = self._base_kp * gain_sched
        self._controller.pi_speed.ki = self._base_ki * gain_sched

        v_d, v_q, theta_e, omega_syn, _ = self._controller.step(
            float(state["t"]),
            float(state["omega_ref"]),
            float(state["omega_m"]),
            _coerce_tuple(state["i_abc"]),
            float(state["torque_e"]),
            float(state["theta_mech"]),
        )

        v_q = _clip(v_q, self._vq_max)
        v_q = _lowpass(v_q, self._vq_prev, self._tau_action, self._dt)
        v_q = _rate_limit(v_q, self._vq_prev, self._dvq_max, self._dt)
        v_q = _clip(v_q, self._vq_max)
        self._vq_prev = v_q
        return PolicyAction(v_d=v_d, v_q=v_q, theta_e=theta_e, omega_syn=omega_syn)


def create_policy(candidate: Mapping[str, object], env: EnvConfig, dt: float) -> object:
    policy_type = str(candidate.get("policy_type", "")).lower()
    params = candidate.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("candidate params must be a dictionary")

    if policy_type in ("vf", "v_f", "scalar"):
        vf_params = _override_scalar_params(env.scalar_vf, params)
        vdc = float(params.get("vdc", env.inverter.Vdc))
        return VfPolicy(vf_params, dt, env.motor.p, vdc)
    if policy_type in ("foc",):
        foc_params = _override_foc_params(env.foc, params)
        return FocPolicy(foc_params, env, dt)
    if policy_type in ("pid_speed", "pid"):
        overrides = dict(params)
        if "kp" in overrides and "kp_speed" not in overrides:
            overrides["kp_speed"] = overrides["kp"]
        if "ki" in overrides and "ki_speed" not in overrides:
            overrides["ki_speed"] = overrides["ki"]
        if "integrator_limit" in overrides and "iq_limit" not in overrides:
            overrides["iq_limit"] = overrides["integrator_limit"]
        foc_params = _override_foc_params(env.foc, overrides)
        return PIDSpeedPolicy(foc_params, env.motor, dt)
    if policy_type in ("mic_rule", "mic"):
        overrides = dict(params)
        if "kp" in overrides and "kp_speed" not in overrides:
            overrides["kp_speed"] = overrides["kp"]
        if "ki" in overrides and "ki_speed" not in overrides:
            overrides["ki_speed"] = overrides["ki"]
        if "integrator_limit" in overrides and "iq_limit" not in overrides:
            overrides["iq_limit"] = overrides["integrator_limit"]
        foc_params = _override_foc_params(env.foc, overrides)
        vq_max = overrides.get("vq_max", foc_params.v_limit)
        if vq_max is None:
            vq_max = env.inverter.Vdc / math.sqrt(3.0)
        else:
            vq_max = float(vq_max)
        dvq_max = overrides.get("dvq_max")
        if dvq_max is not None:
            dvq_max = float(dvq_max)
        tau_action = overrides.get("tau_action")
        if tau_action is not None:
            tau_action = float(tau_action)
        k_age = float(overrides.get("k_age", 0.0))
        return MICRulePolicy(
            foc_params,
            env.motor,
            dt,
            vq_max=vq_max,
            dvq_max=dvq_max,
            tau_action=tau_action,
            k_age=k_age,
        )
    if policy_type in ("nn_stub", "nn"):
        v_d = float(params.get("v_d", 0.0))
        v_q = float(params.get("v_q", 0.0))
        return NnStubPolicy(v_d=v_d, v_q=v_q)
    raise ValueError(f"Unsupported policy_type: {policy_type!r}")


def _override_scalar_params(params: ScalarVfParams, overrides: Mapping[str, object]) -> ScalarVfParams:
    updates: dict[str, float] = {}
    for key in ("k_vf", "u_boost", "f_min", "f_max"):
        if key in overrides:
            updates[key] = float(overrides[key])
    if updates:
        return replace(params, **updates)
    return params


def _override_foc_params(params: FocParams, overrides: Mapping[str, object]) -> FocParams:
    updates: dict[str, float] = {}
    for key in ("kp_id", "ki_id", "kp_iq", "ki_iq", "kp_speed", "ki_speed", "id_ref", "iq_limit", "v_limit"):
        if key in overrides:
            updates[key] = float(overrides[key])
    if updates:
        return replace(params, **updates)
    return params


def _coerce_tuple(value: object) -> tuple[float, float, float]:
    if isinstance(value, Sequence):
        values = list(value)
        if len(values) >= 3:
            return (float(values[0]), float(values[1]), float(values[2]))
    raise ValueError("i_abc must be a sequence of length 3")


def _clip(value: float, limit: float | None) -> float:
    if limit is None:
        return float(value)
    limit_value = float(limit)
    if limit_value <= 0.0:
        return 0.0
    return max(min(float(value), limit_value), -limit_value)


def _rate_limit(value: float, prev: float, rate_limit: float | None, dt: float) -> float:
    if rate_limit is None or rate_limit <= 0.0 or dt <= 0.0:
        return float(value)
    delta_max = rate_limit * dt
    delta = float(value) - float(prev)
    if delta > delta_max:
        return float(prev) + delta_max
    if delta < -delta_max:
        return float(prev) - delta_max
    return float(value)


def _lowpass(value: float, prev: float, tau: float | None, dt: float) -> float:
    if tau is None or tau <= 0.0 or dt <= 0.0:
        return float(value)
    alpha = dt / (tau + dt)
    return float(prev) + alpha * (float(value) - float(prev))


__all__ = [
    "PolicyAction",
    "VfPolicy",
    "FocPolicy",
    "PIDSpeedPolicy",
    "MICRulePolicy",
    "NnStubPolicy",
    "create_policy",
]
