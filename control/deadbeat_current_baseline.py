from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Tuple

from control.safe_neural_horizon_pwm import ControllerStepResult, NeuralTwin
from models.induction_motor_alpha_beta import (
    AlphaBetaInductionMotorModel,
    AlphaBetaMotorParams,
    AlphaBetaMotorState,
)
from models.transformations import alpha_beta_to_dq, dq_to_alpha_beta
from models.two_level_inverter import (
    TwoLevelInverterParams,
    alpha_beta_voltage,
    estimate_inverter_losses,
)
from safety.ai_pwm_gateway import AIPwmRequest, AIPwmSafetyGateway, GateDecision


@dataclass(frozen=True)
class DeadbeatCurrentBaselineConfig:
    dt_s: float = 100.0e-6
    speed_kp: float = 0.040
    speed_ki: float = 0.85
    torque_limit_nm: float = 3.0
    flux_ref_wb: float | None = None
    id_max_fraction: float = 0.45
    iq_max_fraction: float = 0.75
    current_error_weight: float = 1.0
    voltage_error_weight: float = 0.004
    torque_weight: float = 0.10
    flux_weight: float = 0.25
    current_stress_weight: float = 0.22
    switching_weight: float = 1.00
    loss_weight: float = 0.0015
    common_mode_weight: float = 1.0e-4
    voltage_limit_fraction: float = 0.90
    preflux_voltage_fraction: float = 0.32
    feedback_error_threshold_rad_s: float = 6.0
    feedback_uncertainty_threshold: float = 0.40
    temp_trip_c: float = 125.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


class DeadbeatCurrentBaselineController:
    """Host deadbeat predictive current-control baseline with protected vector output.

    The controller computes dq current references from a classical speed/flux
    loop, estimates a deadbeat alpha-beta voltage needed to move the stator
    current to the reference in one PWM tick, and then selects the legal vector
    whose one-step prediction best matches the target current. The Safety
    Gateway remains the only path to gate-level commands.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: DeadbeatCurrentBaselineConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else DeadbeatCurrentBaselineConfig(dt_s=inverter_params.t_pwm_s)
        self.twin = NeuralTwin(motor_params)
        self.speed_integral = 0.0
        self.theta_e = 0.0
        self.last_torque = 0.0
        self.tj_c = float(inverter_params.ambient_c)

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.speed_integral = 0.0
        self.theta_e = 0.0
        self.last_torque = 0.0
        self.tj_c = float(self.inverter_params.ambient_c)

    def _flux_ref(self) -> float:
        if self.cfg.flux_ref_wb is not None and self.cfg.flux_ref_wb > 0.0:
            return float(self.cfg.flux_ref_wb)
        return max(0.02, min(0.45, 0.15 * float(self.motor_params.Lm) * max(self.motor_params.i_limit, 1e-3)))

    def _torque_ref(self, omega_ref: float, omega_hat: float) -> float:
        err = float(omega_ref) - float(omega_hat)
        self.speed_integral = _clamp(self.speed_integral + err * float(self.cfg.dt_s), -10.0, 10.0)
        raw = float(self.cfg.speed_kp) * err + float(self.cfg.speed_ki) * self.speed_integral
        return _clamp(raw, -float(self.cfg.torque_limit_nm), float(self.cfg.torque_limit_nm))

    def _update_angle(self, state: AlphaBetaMotorState, omega_ref: float, torque_ref: float) -> None:
        flux_alpha, flux_beta = state.psi_r_alpha, state.psi_r_beta
        flux_abs = math.hypot(flux_alpha, flux_beta)
        if flux_abs < 1.0e-5:
            flux_alpha, flux_beta = state.psi_s_alpha, state.psi_s_beta
            flux_abs = math.hypot(flux_alpha, flux_beta)

        flux_ref = self._flux_ref()
        slip = 0.0
        if flux_ref > 1.0e-6:
            slip = _clamp(torque_ref / (1.5 * max(self.motor_params.p, 1) * flux_ref), -120.0, 120.0)
        self.theta_e = _wrap_angle(
            self.theta_e + (float(self.motor_params.p) * float(omega_ref) + slip) * float(self.cfg.dt_s)
        )
        if flux_abs > 1.0e-4:
            measured = math.atan2(flux_beta, flux_alpha)
            x = 0.72 * math.cos(self.theta_e) + 0.28 * math.cos(measured)
            y = 0.72 * math.sin(self.theta_e) + 0.28 * math.sin(measured)
            self.theta_e = math.atan2(y, x)

    def _dq_references(self, torque_ref: float) -> Tuple[float, float]:
        flux_ref = self._flux_ref()
        id_limit = max(0.05, float(self.cfg.id_max_fraction) * max(self.motor_params.i_limit, 1e-6))
        iq_limit = max(0.05, float(self.cfg.iq_max_fraction) * max(self.motor_params.i_limit, 1e-6))
        id_ref = _clamp(flux_ref / max(float(self.motor_params.Lm), 1e-9), 0.0, id_limit)
        iq_ref = _clamp(
            torque_ref / max(1.5 * max(self.motor_params.p, 1) * max(flux_ref, 1e-6), 1e-9),
            -iq_limit,
            iq_limit,
        )
        return id_ref, iq_ref

    def _deadbeat_voltage_ref(
        self,
        *,
        i_alpha: float,
        i_beta: float,
        i_alpha_ref: float,
        i_beta_ref: float,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
    ) -> tuple[float, float]:
        ls = float(self.motor_params.Lls) + float(self.motor_params.Lm)
        rs = float(self.motor_params.Rs) * (
            1.0 + float(self.motor_params.rs_temp_coeff) * (float(state.temp_s_c) - float(self.motor_params.temp_ref_c))
        )
        dt = max(float(self.cfg.dt_s), 1e-9)
        v_alpha = rs * float(i_alpha) + ls * (float(i_alpha_ref) - float(i_alpha)) / dt
        v_beta = rs * float(i_beta) + ls * (float(i_beta_ref) - float(i_beta)) / dt

        flux_ref = self._flux_ref()
        flux_abs = math.hypot(state.psi_s_alpha, state.psi_s_beta)
        if flux_abs < 0.55 * flux_ref:
            preflux_v = float(self.cfg.preflux_voltage_fraction) * abs(float(inverter.Vdc))
            v_alpha += preflux_v * math.cos(self.theta_e)
            v_beta += preflux_v * math.sin(self.theta_e)

        limit = max(1.0, float(self.cfg.voltage_limit_fraction) * abs(float(inverter.Vdc)) / math.sqrt(3.0))
        mag = math.hypot(v_alpha, v_beta)
        if mag > limit:
            scale = limit / mag
            v_alpha *= scale
            v_beta *= scale
        return float(v_alpha), float(v_beta)

    def _score_vector(
        self,
        *,
        vector_id: int,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
        load_torque_nm: float,
        currents_alpha_beta: Tuple[float, float],
        i_alpha_ref: float,
        i_beta_ref: float,
        v_alpha_ref: float,
        v_beta_ref: float,
        torque_ref: float,
    ) -> tuple[float, Dict[str, float]]:
        v_alpha, v_beta = alpha_beta_voltage(vector_id, inverter, i_alpha_beta=currents_alpha_beta)
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        predicted = model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=state)
        losses = estimate_inverter_losses(
            prev_vector_id=self.gateway.current_vector_id,
            next_vector_id=vector_id,
            params=inverter,
            i_alpha_beta=currents_alpha_beta,
        )
        current_error = math.hypot(
            predicted.currents.i_s_alpha - float(i_alpha_ref),
            predicted.currents.i_s_beta - float(i_beta_ref),
        )
        voltage_error = math.hypot(v_alpha - float(v_alpha_ref), v_beta - float(v_beta_ref))
        flux_abs = math.hypot(predicted.state.psi_s_alpha, predicted.state.psi_s_beta)
        flux_error = abs(self._flux_ref() - flux_abs)
        torque_error = abs(float(torque_ref) - float(predicted.torque_nm))
        current_ratio = predicted.currents.stator_abs / max(float(self.motor_params.i_limit), 1e-9)
        current_stress = predicted.currents.stator_abs + 3.0 * max(0.0, current_ratio - 0.88) ** 2
        cost = (
            float(self.cfg.current_error_weight) * current_error
            + float(self.cfg.voltage_error_weight) * voltage_error
            + float(self.cfg.torque_weight) * torque_error
            + float(self.cfg.flux_weight) * flux_error
            + float(self.cfg.current_stress_weight) * current_stress
            + float(self.cfg.switching_weight) * losses.switch_events
            + float(self.cfg.loss_weight) * losses.total_w
            + float(self.cfg.common_mode_weight) * abs(losses.common_mode_v)
        )
        return float(cost), {
            "candidate_cost": float(cost),
            "candidate_current_error": float(current_error),
            "candidate_voltage_error": float(voltage_error),
            "candidate_current": float(predicted.currents.stator_abs),
            "candidate_flux": float(flux_abs),
            "candidate_torque": float(predicted.torque_nm),
            "candidate_switch_events": float(losses.switch_events),
            "candidate_loss_w": float(losses.total_w),
        }

    def _select_vector(
        self,
        *,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
        load_torque_nm: float,
        currents_alpha_beta: Tuple[float, float],
        i_alpha_ref: float,
        i_beta_ref: float,
        v_alpha_ref: float,
        v_beta_ref: float,
        torque_ref: float,
    ) -> tuple[int, Dict[str, float]]:
        best_vector = self.gateway.current_vector_id
        best_cost = float("inf")
        best_metrics: Dict[str, float] = {}
        for vector_id in range(8):
            cost, metrics = self._score_vector(
                vector_id=vector_id,
                state=state,
                inverter=inverter,
                load_torque_nm=load_torque_nm,
                currents_alpha_beta=currents_alpha_beta,
                i_alpha_ref=i_alpha_ref,
                i_beta_ref=i_beta_ref,
                v_alpha_ref=v_alpha_ref,
                v_beta_ref=v_beta_ref,
                torque_ref=torque_ref,
            )
            if cost < best_cost:
                best_cost = cost
                best_vector = int(vector_id)
                best_metrics = metrics
        best_metrics["cost"] = float(best_cost)
        return best_vector, best_metrics

    def step(
        self,
        *,
        omega_ref: float,
        load_torque_nm: float,
        measured_state: AlphaBetaMotorState | None = None,
        measured_i_abs: float = 0.0,
        vdc: float | None = None,
    ) -> ControllerStepResult:
        if measured_state is not None:
            self.twin.correct(measured_state, alpha=1.0)
        else:
            self.twin.drift_without_feedback()

        inverter = replace(self.inverter_params, Vdc=float(self.inverter_params.Vdc if vdc is None else vdc))
        state = self.twin.state_hat
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        currents = model.currents()
        torque_ref = self._torque_ref(omega_ref, state.omega_m)
        self._update_angle(state, omega_ref, torque_ref)
        id_ref, iq_ref = self._dq_references(torque_ref)
        i_alpha_ref, i_beta_ref = dq_to_alpha_beta(id_ref, iq_ref, self.theta_e)
        i_d, i_q = alpha_beta_to_dq(currents.i_s_alpha, currents.i_s_beta, self.theta_e)
        v_alpha_ref, v_beta_ref = self._deadbeat_voltage_ref(
            i_alpha=currents.i_s_alpha,
            i_beta=currents.i_s_beta,
            i_alpha_ref=i_alpha_ref,
            i_beta_ref=i_beta_ref,
            state=state,
            inverter=inverter,
        )
        vector_id, metrics = self._select_vector(
            state=state,
            inverter=inverter,
            load_torque_nm=load_torque_nm,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            i_alpha_ref=i_alpha_ref,
            i_beta_ref=i_beta_ref,
            v_alpha_ref=v_alpha_ref,
            v_beta_ref=v_beta_ref,
            torque_ref=torque_ref,
        )
        predicted_state, predicted_torque, predicted_i_abs = self.twin.predict(
            vector_id,
            inverter,
            load_torque_nm,
            self.cfg.dt_s,
        )
        current_ratio = predicted_i_abs / max(self.motor_params.i_limit, 1e-9)
        predicted_risk = max(0.0, current_ratio - 0.85) + 0.20 * self.twin.uncertainty
        confidence = max(0.52, min(0.99, self.twin.confidence()))
        prev_vector_id = self.gateway.current_vector_id
        decision: GateDecision = self.gateway.evaluate(
            AIPwmRequest(
                vector_id=vector_id,
                dwell_s=self.cfg.dt_s,
                confidence=confidence,
                predicted_i_abs=predicted_i_abs,
                measured_i_abs=measured_i_abs,
                vdc=inverter.Vdc,
                tj_c=self.tj_c,
                predicted_risk=predicted_risk,
            )
        )
        applied_vector = decision.vector_id if decision.pwm_enabled else 0
        if decision.pwm_enabled:
            losses = estimate_inverter_losses(
                prev_vector_id=prev_vector_id,
                next_vector_id=applied_vector,
                params=inverter,
                i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            )
            applied_loss_w = float(losses.total_w)
            applied_switch_events = float(losses.switch_events)
        else:
            applied_loss_w = 0.0
            applied_switch_events = 0.0

        applied_state, applied_torque, _ = self.twin.predict(applied_vector, inverter, load_torque_nm, self.cfg.dt_s)
        self.twin.state_hat = applied_state
        self.last_torque = applied_torque
        tau = max(inverter.thermal_rth_k_per_w * inverter.thermal_cth_j_per_k, 1e-9)
        target_temp = inverter.ambient_c + inverter.thermal_rth_k_per_w * applied_loss_w
        self.tj_c += (target_temp - self.tj_c) * min(1.0, self.cfg.dt_s / tau)

        metrics = dict(metrics)
        metrics.update(
            {
                "id_ref": float(id_ref),
                "iq_ref": float(iq_ref),
                "i_d": float(i_d),
                "i_q": float(i_q),
                "i_alpha_ref": float(i_alpha_ref),
                "i_beta_ref": float(i_beta_ref),
                "v_alpha_deadbeat": float(v_alpha_ref),
                "v_beta_deadbeat": float(v_beta_ref),
                "torque_ref": float(torque_ref),
                "predicted_torque": float(predicted_torque),
                "applied_torque": float(applied_torque),
                "loss_w": applied_loss_w,
                "switch_events": applied_switch_events,
                "accepted": 1.0 if decision.accepted else 0.0,
                "fault_flags": float(int(decision.fault_flags)),
                "tj_c": float(self.tj_c),
            }
        )
        return ControllerStepResult(
            decision=decision,
            vector_id=applied_vector,
            feedback_requested=measured_state is not None,
            confidence=confidence,
            predicted_i_abs=float(predicted_i_abs),
            predicted_risk=float(predicted_risk),
            metrics=metrics,
        )


__all__ = [
    "DeadbeatCurrentBaselineConfig",
    "DeadbeatCurrentBaselineController",
]
