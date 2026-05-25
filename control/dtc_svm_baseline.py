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
class DtcSvmBaselineConfig:
    dt_s: float = 100.0e-6
    speed_kp: float = 0.045
    speed_ki: float = 0.95
    torque_limit_nm: float = 3.0
    flux_ref_wb: float | None = None
    torque_voltage_kp: float = 22.0
    torque_voltage_ki: float = 480.0
    flux_voltage_kp: float = 190.0
    flux_voltage_ki: float = 2800.0
    voltage_limit_fraction: float = 0.88
    current_weight: float = 0.20
    switching_weight: float = 0.030
    loss_weight: float = 0.002
    common_mode_weight: float = 1.0e-4
    zero_vector_flux_penalty: float = 0.45
    preflux_voltage_fraction: float = 0.35
    feedback_error_threshold_rad_s: float = 6.0
    feedback_uncertainty_threshold: float = 0.40
    temp_trip_c: float = 125.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


class DtcSvmBaselineController:
    """Host DTC-SVM baseline with PI torque/flux loops and protected key output.

    This is a classical host baseline, not neural control. Torque and stator-flux
    errors synthesize an alpha-beta voltage reference in the stator-flux frame;
    the SVM stage is represented by nearest legal-vector selection because the
    current plant accepts one inverter vector per PWM tick. The selected vector
    still passes through the same Safety Gateway as SNH-PWM.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: DtcSvmBaselineConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else DtcSvmBaselineConfig(dt_s=inverter_params.t_pwm_s)
        self.twin = NeuralTwin(motor_params)
        self.speed_integral = 0.0
        self.torque_integral = 0.0
        self.flux_integral = 0.0
        self.theta_probe = 0.0
        self.last_torque = 0.0
        self.tj_c = float(inverter_params.ambient_c)

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.speed_integral = 0.0
        self.torque_integral = 0.0
        self.flux_integral = 0.0
        self.theta_probe = 0.0
        self.last_torque = 0.0
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

    def _flux_frame(self, state: AlphaBetaMotorState, omega_ref: float) -> tuple[float, float, float]:
        flux_abs = math.hypot(state.psi_s_alpha, state.psi_s_beta)
        if flux_abs > 1.0e-5:
            theta = math.atan2(state.psi_s_beta, state.psi_s_alpha)
        else:
            self.theta_probe = _wrap_angle(self.theta_probe + float(self.motor_params.p) * float(omega_ref) * self.cfg.dt_s)
            theta = self.theta_probe
        return math.cos(theta), math.sin(theta), flux_abs

    def _voltage_reference(
        self,
        *,
        state: AlphaBetaMotorState,
        torque_ref: float,
        torque_now: float,
        flux_ref: float,
        flux_now: float,
        omega_ref: float,
        inverter: TwoLevelInverterParams,
    ) -> Dict[str, float]:
        flux_u_alpha, flux_u_beta, _ = self._flux_frame(state, omega_ref)
        torque_error = float(torque_ref) - float(torque_now)
        flux_error = float(flux_ref) - float(flux_now)
        self.torque_integral = _clamp(self.torque_integral + torque_error * self.cfg.dt_s, -2.0, 2.0)
        self.flux_integral = _clamp(self.flux_integral + flux_error * self.cfg.dt_s, -0.2, 0.2)

        v_torque = float(self.cfg.torque_voltage_kp) * torque_error + float(self.cfg.torque_voltage_ki) * self.torque_integral
        v_flux = float(self.cfg.flux_voltage_kp) * flux_error + float(self.cfg.flux_voltage_ki) * self.flux_integral
        tangent_alpha = -flux_u_beta
        tangent_beta = flux_u_alpha
        v_alpha_ref = v_flux * flux_u_alpha + v_torque * tangent_alpha
        v_beta_ref = v_flux * flux_u_beta + v_torque * tangent_beta

        if flux_now < 0.45 * flux_ref:
            preflux_v = float(self.cfg.preflux_voltage_fraction) * abs(float(inverter.Vdc))
            v_alpha_ref += preflux_v * flux_u_alpha
            v_beta_ref += preflux_v * flux_u_beta

        limit = max(1.0, float(self.cfg.voltage_limit_fraction) * abs(float(inverter.Vdc)) / math.sqrt(3.0))
        mag = math.hypot(v_alpha_ref, v_beta_ref)
        if mag > limit:
            scale = limit / mag
            v_alpha_ref *= scale
            v_beta_ref *= scale
        return {
            "v_alpha_ref": float(v_alpha_ref),
            "v_beta_ref": float(v_beta_ref),
            "v_ref_abs": float(math.hypot(v_alpha_ref, v_beta_ref)),
            "torque_error": float(torque_error),
            "flux_error": float(flux_error),
            "v_torque_axis": float(v_torque),
            "v_flux_axis": float(v_flux),
        }

    def _score_vector(
        self,
        *,
        vector_id: int,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
        v_alpha_ref: float,
        v_beta_ref: float,
        load_torque_nm: float,
        currents_alpha_beta: Tuple[float, float],
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
        voltage_error = (v_alpha - v_alpha_ref) ** 2 + (v_beta - v_beta_ref) ** 2
        current_ratio = predicted.currents.stator_abs / max(float(self.motor_params.i_limit), 1e-9)
        current_cost = predicted.currents.stator_abs + 3.0 * max(0.0, current_ratio - 0.88) ** 2
        flux_abs = math.hypot(predicted.state.psi_s_alpha, predicted.state.psi_s_beta)
        if flux_abs < 0.7 * self._flux_ref() and vector_id in (0, 7):
            voltage_error += (float(self.cfg.zero_vector_flux_penalty) * max(abs(inverter.Vdc), 1.0)) ** 2
        cost = (
            voltage_error
            + float(self.cfg.current_weight) * current_cost * max(abs(inverter.Vdc), 1.0)
            + float(self.cfg.switching_weight) * losses.switch_events * max(abs(inverter.Vdc), 1.0)
            + float(self.cfg.loss_weight) * losses.total_w
            + float(self.cfg.common_mode_weight) * abs(losses.common_mode_v)
        )
        return float(cost), {
            "candidate_cost": float(cost),
            "candidate_current": float(predicted.currents.stator_abs),
            "candidate_flux": float(flux_abs),
            "candidate_torque": float(predicted.torque_nm),
            "candidate_switch_events": float(losses.switch_events),
            "candidate_loss_w": float(losses.total_w),
            "candidate_voltage_error": float(voltage_error),
        }

    def _select_svm_vector(
        self,
        *,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
        v_alpha_ref: float,
        v_beta_ref: float,
        load_torque_nm: float,
        currents_alpha_beta: Tuple[float, float],
    ) -> tuple[int, Dict[str, float]]:
        best_vector = self.gateway.current_vector_id
        best_cost = float("inf")
        best_metrics: Dict[str, float] = {}
        for vector_id in range(8):
            cost, metrics = self._score_vector(
                vector_id=vector_id,
                state=state,
                inverter=inverter,
                v_alpha_ref=v_alpha_ref,
                v_beta_ref=v_beta_ref,
                load_torque_nm=load_torque_nm,
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
        torque_now = model.torque_nm(state, currents)
        flux_now = math.hypot(state.psi_s_alpha, state.psi_s_beta)
        flux_ref = self._flux_ref()
        torque_ref = self._torque_ref(omega_ref, state.omega_m)
        vref = self._voltage_reference(
            state=state,
            torque_ref=torque_ref,
            torque_now=torque_now,
            flux_ref=flux_ref,
            flux_now=flux_now,
            omega_ref=omega_ref,
            inverter=inverter,
        )
        vector_id, metrics = self._select_svm_vector(
            state=state,
            inverter=inverter,
            v_alpha_ref=vref["v_alpha_ref"],
            v_beta_ref=vref["v_beta_ref"],
            load_torque_nm=load_torque_nm,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
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
        metrics.update(vref)
        metrics.update(
            {
                "torque_ref": float(torque_ref),
                "torque_now": float(torque_now),
                "flux_ref": float(flux_ref),
                "flux_now": float(flux_now),
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
    "DtcSvmBaselineConfig",
    "DtcSvmBaselineController",
]
