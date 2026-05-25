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
from models.two_level_inverter import (
    TwoLevelInverterParams,
    alpha_beta_voltage,
    estimate_inverter_losses,
)
from safety.ai_pwm_gateway import AIPwmRequest, AIPwmSafetyGateway, GateDecision


@dataclass(frozen=True)
class FcsMpcOneStepBaselineConfig:
    dt_s: float = 100.0e-6
    speed_kp: float = 0.040
    speed_ki: float = 1.20
    torque_limit_nm: float = 3.5
    flux_ref_wb: float | None = None
    speed_weight: float = 0.20
    torque_weight: float = 1.00
    flux_weight: float = 0.85
    current_weight: float = 0.16
    switching_weight: float = 0.030
    loss_weight: float = 0.002
    torque_ripple_weight: float = 0.05
    common_mode_weight: float = 1.0e-4
    zero_vector_flux_penalty: float = 0.40
    feedback_error_threshold_rad_s: float = 5.0
    feedback_uncertainty_threshold: float = 0.35
    temp_trip_c: float = 125.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


class FcsMpcOneStepBaselineController:
    """Host one-step FCS-MPC baseline over the same 8 legal inverter vectors.

    This controller is deliberately non-neural: it computes a speed-loop torque
    reference, predicts one PWM step for each legal vector, and selects the vector
    with the lowest current/torque/flux/switching/loss cost before passing it
    through the same Safety Gateway as SNH-PWM.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: FcsMpcOneStepBaselineConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else FcsMpcOneStepBaselineConfig(dt_s=inverter_params.t_pwm_s)
        self.twin = NeuralTwin(motor_params)
        self.speed_integral = 0.0
        self.last_torque = 0.0
        self.theta_probe = 0.0
        self.tj_c = float(inverter_params.ambient_c)

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.speed_integral = 0.0
        self.last_torque = 0.0
        self.theta_probe = 0.0
        self.tj_c = float(self.inverter_params.ambient_c)

    def _flux_ref(self) -> float:
        if self.cfg.flux_ref_wb is not None and self.cfg.flux_ref_wb > 0.0:
            return float(self.cfg.flux_ref_wb)
        return max(0.02, min(0.45, 0.15 * float(self.motor_params.Lm) * max(self.motor_params.i_limit, 1e-3)))

    def _torque_ref(self, omega_ref: float, omega_hat: float) -> float:
        err = float(omega_ref) - float(omega_hat)
        self.speed_integral = _clamp(self.speed_integral + err * float(self.cfg.dt_s), -12.0, 12.0)
        raw = float(self.cfg.speed_kp) * err + float(self.cfg.speed_ki) * self.speed_integral
        return _clamp(raw, -float(self.cfg.torque_limit_nm), float(self.cfg.torque_limit_nm))

    def _score_vector(
        self,
        *,
        vector_id: int,
        state: AlphaBetaMotorState,
        omega_ref: float,
        torque_ref: float,
        load_torque_nm: float,
        inverter: TwoLevelInverterParams,
        currents_alpha_beta: Tuple[float, float],
    ) -> tuple[float, Dict[str, float], AlphaBetaMotorState]:
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        v_alpha, v_beta = alpha_beta_voltage(vector_id, inverter, i_alpha_beta=currents_alpha_beta)
        step = model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=state)
        losses = estimate_inverter_losses(
            prev_vector_id=self.gateway.current_vector_id,
            next_vector_id=vector_id,
            params=inverter,
            i_alpha_beta=currents_alpha_beta,
        )
        flux_abs = math.hypot(step.state.psi_s_alpha, step.state.psi_s_beta)
        flux_error = abs(self._flux_ref() - flux_abs)
        if flux_abs < 0.8 * self._flux_ref() and vector_id in (0, 7):
            flux_error += float(self.cfg.zero_vector_flux_penalty)
        current_ratio = step.currents.stator_abs / max(float(self.motor_params.i_limit), 1e-9)
        current_penalty = step.currents.stator_abs + 2.0 * max(0.0, current_ratio - 0.85) ** 2
        torque_ripple = abs(step.torque_nm - self.last_torque)
        speed_error = abs(float(omega_ref) - step.state.omega_m)
        cost = (
            float(self.cfg.speed_weight) * speed_error
            + float(self.cfg.torque_weight) * abs(torque_ref - step.torque_nm)
            + float(self.cfg.flux_weight) * flux_error
            + float(self.cfg.current_weight) * current_penalty
            + float(self.cfg.switching_weight) * losses.switch_events
            + float(self.cfg.loss_weight) * losses.total_w
            + float(self.cfg.torque_ripple_weight) * torque_ripple
            + float(self.cfg.common_mode_weight) * abs(losses.common_mode_v)
        )
        metrics = {
            "candidate_cost": float(cost),
            "candidate_torque": float(step.torque_nm),
            "candidate_current": float(step.currents.stator_abs),
            "candidate_flux": float(flux_abs),
            "candidate_switch_events": float(losses.switch_events),
            "candidate_loss_w": float(losses.total_w),
            "candidate_common_mode_v": float(losses.common_mode_v),
            "candidate_speed_error": float(speed_error),
        }
        return float(cost), metrics, step.state

    def _select_vector(
        self,
        *,
        state: AlphaBetaMotorState,
        omega_ref: float,
        torque_ref: float,
        load_torque_nm: float,
        inverter: TwoLevelInverterParams,
        currents_alpha_beta: Tuple[float, float],
    ) -> tuple[int, Dict[str, float]]:
        best_vector = self.gateway.current_vector_id
        best_cost = float("inf")
        best_metrics: Dict[str, float] = {}
        for vector_id in range(8):
            cost, metrics, _ = self._score_vector(
                vector_id=vector_id,
                state=state,
                omega_ref=omega_ref,
                torque_ref=torque_ref,
                load_torque_nm=load_torque_nm,
                inverter=inverter,
                currents_alpha_beta=currents_alpha_beta,
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
        self.theta_probe = (self.theta_probe + float(self.motor_params.p) * float(omega_ref) * self.cfg.dt_s) % (
            2.0 * math.pi
        )
        vector_id, metrics = self._select_vector(
            state=state,
            omega_ref=omega_ref,
            torque_ref=torque_ref,
            load_torque_nm=load_torque_nm,
            inverter=inverter,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
        )
        predicted_state, predicted_torque, predicted_i_abs = self.twin.predict(
            vector_id,
            inverter,
            load_torque_nm,
            self.cfg.dt_s,
        )
        predicted_risk = max(0.0, predicted_i_abs / max(self.motor_params.i_limit, 1e-9) - 0.85)
        confidence = max(0.55, min(0.99, self.twin.confidence()))
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
    "FcsMpcOneStepBaselineConfig",
    "FcsMpcOneStepBaselineController",
]
