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
class SensorlessAdaptiveFocBaselineConfig:
    dt_s: float = 100.0e-6
    speed_kp: float = 0.032
    speed_ki: float = 0.70
    current_kp: float = 18.0
    current_ki: float = 520.0
    torque_limit_nm: float = 2.8
    flux_ref_wb: float | None = None
    id_max_fraction: float = 0.42
    iq_max_fraction: float = 0.70
    voltage_limit_fraction: float = 0.86
    switching_tiebreak_weight: float = 0.040
    preflux_voltage_fraction: float = 0.42
    observer_gain: float = 0.18
    observer_max_accel_rad_s2: float = 4500.0
    min_flux_for_observer_wb: float = 0.015
    rs_adaptation_gain: float = 0.010
    rs_scale_min: float = 0.55
    rs_scale_max: float = 1.85
    feedback_error_threshold_rad_s: float = 36.0
    feedback_uncertainty_threshold: float = 0.85
    temp_trip_c: float = 125.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


class SensorlessAdaptiveFocBaselineController:
    """Host sensorless/adaptive FOC baseline with protected vector output.

    This baseline does not use measured mechanical speed as the speed feedback
    variable. When a measured state is available in host simulation, only flux
    and current consistency are used to update a lightweight MRAS-like speed
    observer and Rs adaptation. The fast command path remains FOC/SVM-style
    nearest legal-vector selection through the same Safety Gateway.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: SensorlessAdaptiveFocBaselineConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else SensorlessAdaptiveFocBaselineConfig(dt_s=inverter_params.t_pwm_s)
        self.twin = NeuralTwin(motor_params)
        self.speed_integral = 0.0
        self.id_integral = 0.0
        self.iq_integral = 0.0
        self.theta_e = 0.0
        self.last_flux_angle: float | None = None
        self.omega_hat = 0.0
        self.rs_scale = 1.0
        self.last_torque = 0.0
        self.tj_c = float(inverter_params.ambient_c)

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.speed_integral = 0.0
        self.id_integral = 0.0
        self.iq_integral = 0.0
        self.theta_e = 0.0
        self.last_flux_angle = None
        self.omega_hat = float(state.omega_m) if state is not None else 0.0
        self.rs_scale = 1.0
        self.last_torque = 0.0
        self.tj_c = float(self.inverter_params.ambient_c)

    def _adaptive_params(self) -> AlphaBetaMotorParams:
        return replace(self.motor_params, Rs=max(1e-9, float(self.motor_params.Rs) * float(self.rs_scale)))

    def _flux_ref(self) -> float:
        if self.cfg.flux_ref_wb is not None and self.cfg.flux_ref_wb > 0.0:
            return float(self.cfg.flux_ref_wb)
        return max(0.02, min(0.45, 0.15 * float(self.motor_params.Lm) * max(self.motor_params.i_limit, 1e-3)))

    def _mras_speed_update(self, measured_state: AlphaBetaMotorState | None) -> None:
        if measured_state is None:
            self.omega_hat = self.twin.state_hat.omega_m
            return
        flux_alpha = measured_state.psi_r_alpha
        flux_beta = measured_state.psi_r_beta
        flux_abs = math.hypot(flux_alpha, flux_beta)
        if flux_abs < float(self.cfg.min_flux_for_observer_wb):
            flux_alpha = measured_state.psi_s_alpha
            flux_beta = measured_state.psi_s_beta
            flux_abs = math.hypot(flux_alpha, flux_beta)
        if flux_abs < float(self.cfg.min_flux_for_observer_wb):
            return
        angle = math.atan2(flux_beta, flux_alpha)
        if self.last_flux_angle is not None:
            dtheta = _wrap_angle(angle - self.last_flux_angle)
            raw_omega = dtheta / (max(self.motor_params.p, 1) * max(float(self.cfg.dt_s), 1e-9))
            max_delta = float(self.cfg.observer_max_accel_rad_s2) * float(self.cfg.dt_s)
            target = self.omega_hat + _clamp(raw_omega - self.omega_hat, -max_delta, max_delta)
            self.omega_hat += float(self.cfg.observer_gain) * (target - self.omega_hat)
        self.last_flux_angle = angle

    def _adapt_rs(self, measured_state: AlphaBetaMotorState | None) -> None:
        if measured_state is None:
            return
        measured_model = AlphaBetaInductionMotorModel(self.motor_params, measured_state)
        estimated_model = AlphaBetaInductionMotorModel(self._adaptive_params(), self.twin.state_hat)
        measured_i = measured_model.currents(measured_state).stator_abs
        estimated_i = estimated_model.currents(self.twin.state_hat).stator_abs
        residual = (measured_i - estimated_i) / max(float(self.motor_params.i_limit), 1e-9)
        self.rs_scale = _clamp(
            self.rs_scale + float(self.cfg.rs_adaptation_gain) * residual,
            float(self.cfg.rs_scale_min),
            float(self.cfg.rs_scale_max),
        )

    def _sensorless_correct(self, measured_state: AlphaBetaMotorState | None) -> None:
        self._mras_speed_update(measured_state)
        self._adapt_rs(measured_state)
        if measured_state is None:
            self.twin.drift_without_feedback()
            self.twin.state_hat = replace(self.twin.state_hat, omega_m=float(self.omega_hat))
            return
        sensorless_state = replace(measured_state, omega_m=float(self.omega_hat))
        self.twin.correct(sensorless_state, alpha=0.55)
        self.twin.state_hat = replace(self.twin.state_hat, omega_m=float(self.omega_hat))

    def _torque_ref(self, omega_ref: float) -> float:
        err = float(omega_ref) - float(self.omega_hat)
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
            slip = _clamp(torque_ref / (1.5 * max(self.motor_params.p, 1) * flux_ref), -100.0, 100.0)
        self.theta_e = _wrap_angle(self.theta_e + (float(self.motor_params.p) * self.omega_hat + slip) * self.cfg.dt_s)
        if flux_abs > 1.0e-4:
            measured = math.atan2(flux_beta, flux_alpha)
            x = 0.68 * math.cos(self.theta_e) + 0.32 * math.cos(measured)
            y = 0.68 * math.sin(self.theta_e) + 0.32 * math.sin(measured)
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

    def _voltage_reference(
        self,
        *,
        id_ref: float,
        iq_ref: float,
        i_d: float,
        i_q: float,
        inverter: TwoLevelInverterParams,
    ) -> Tuple[float, float]:
        err_d = float(id_ref) - float(i_d)
        err_q = float(iq_ref) - float(i_q)
        self.id_integral = _clamp(self.id_integral + err_d * float(self.cfg.dt_s), -1.2, 1.2)
        self.iq_integral = _clamp(self.iq_integral + err_q * float(self.cfg.dt_s), -1.2, 1.2)
        ls = float(self.motor_params.Lls) + float(self.motor_params.Lm)
        omega_e = float(self.motor_params.p) * float(self.omega_hat)
        v_d = float(self.cfg.current_kp) * err_d + float(self.cfg.current_ki) * self.id_integral - omega_e * ls * i_q
        v_q = float(self.cfg.current_kp) * err_q + float(self.cfg.current_ki) * self.iq_integral + omega_e * ls * i_d

        v_alpha, v_beta = dq_to_alpha_beta(v_d, v_q, self.theta_e)
        limit = max(1.0, float(self.cfg.voltage_limit_fraction) * abs(float(inverter.Vdc)) / math.sqrt(3.0))
        mag = math.hypot(v_alpha, v_beta)
        if mag > limit:
            scale = limit / mag
            v_alpha *= scale
            v_beta *= scale
        return float(v_alpha), float(v_beta)

    def _select_svm_vector(
        self,
        *,
        v_alpha_ref: float,
        v_beta_ref: float,
        currents_alpha_beta: Tuple[float, float],
        inverter: TwoLevelInverterParams,
    ) -> int:
        scored: list[tuple[float, int]] = []
        prev = self.gateway.current_vector_id
        for vector_id in range(8):
            v_alpha, v_beta = alpha_beta_voltage(vector_id, inverter, i_alpha_beta=currents_alpha_beta)
            dist = (v_alpha - v_alpha_ref) ** 2 + (v_beta - v_beta_ref) ** 2
            losses = estimate_inverter_losses(
                prev_vector_id=prev,
                next_vector_id=vector_id,
                params=inverter,
                i_alpha_beta=currents_alpha_beta,
            )
            cost = dist + float(self.cfg.switching_tiebreak_weight) * losses.switch_events * max(abs(inverter.Vdc), 1.0)
            scored.append((cost, vector_id))
        scored.sort(key=lambda item: item[0])
        return int(scored[0][1])

    def _predict_step(
        self,
        *,
        vector_id: int,
        state: AlphaBetaMotorState,
        inverter: TwoLevelInverterParams,
        load_torque_nm: float,
        currents_alpha_beta: Tuple[float, float],
    ):
        v_alpha, v_beta = alpha_beta_voltage(vector_id, inverter, i_alpha_beta=currents_alpha_beta)
        model = AlphaBetaInductionMotorModel(self._adaptive_params(), state)
        return model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=state)

    def step(
        self,
        *,
        omega_ref: float,
        load_torque_nm: float,
        measured_state: AlphaBetaMotorState | None = None,
        measured_i_abs: float = 0.0,
        vdc: float | None = None,
    ) -> ControllerStepResult:
        self._sensorless_correct(measured_state)

        inverter = replace(self.inverter_params, Vdc=float(self.inverter_params.Vdc if vdc is None else vdc))
        state = self.twin.state_hat
        model = AlphaBetaInductionMotorModel(self._adaptive_params(), state)
        currents = model.currents()
        torque_ref = self._torque_ref(omega_ref)
        self._update_angle(state, omega_ref, torque_ref)
        i_d, i_q = alpha_beta_to_dq(currents.i_s_alpha, currents.i_s_beta, self.theta_e)
        id_ref, iq_ref = self._dq_references(torque_ref)
        v_alpha_ref, v_beta_ref = self._voltage_reference(
            id_ref=id_ref,
            iq_ref=iq_ref,
            i_d=i_d,
            i_q=i_q,
            inverter=inverter,
        )
        flux_ref = self._flux_ref()
        flux_abs = math.hypot(state.psi_s_alpha, state.psi_s_beta)
        if flux_abs < 0.65 * flux_ref:
            v_pre = float(self.cfg.preflux_voltage_fraction) * abs(float(inverter.Vdc))
            v_alpha_ref += v_pre * math.cos(self.theta_e)
            v_beta_ref += v_pre * math.sin(self.theta_e)
        vector_id = self._select_svm_vector(
            v_alpha_ref=v_alpha_ref,
            v_beta_ref=v_beta_ref,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            inverter=inverter,
        )
        predicted = self._predict_step(
            vector_id=vector_id,
            state=state,
            inverter=inverter,
            load_torque_nm=load_torque_nm,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
        )
        predicted_i_abs = predicted.currents.stator_abs
        current_ratio = predicted_i_abs / max(self.motor_params.i_limit, 1e-9)
        predicted_risk = max(0.0, current_ratio - 0.85) + 0.25 * self.twin.uncertainty
        confidence = max(0.42, min(0.96, self.twin.confidence() - 0.05 * abs(self.rs_scale - 1.0)))
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

        applied = self._predict_step(
            vector_id=applied_vector,
            state=state,
            inverter=inverter,
            load_torque_nm=load_torque_nm,
            currents_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
        )
        self.twin.state_hat = replace(applied.state, omega_m=float(self.omega_hat))
        self.last_torque = float(applied.torque_nm)
        tau = max(inverter.thermal_rth_k_per_w * inverter.thermal_cth_j_per_k, 1e-9)
        target_temp = inverter.ambient_c + inverter.thermal_rth_k_per_w * applied_loss_w
        self.tj_c += (target_temp - self.tj_c) * min(1.0, self.cfg.dt_s / tau)

        metrics: Dict[str, float] = {
            "id_ref": float(id_ref),
            "iq_ref": float(iq_ref),
            "i_d": float(i_d),
            "i_q": float(i_q),
            "omega_hat": float(self.omega_hat),
            "rs_scale": float(self.rs_scale),
            "torque_ref": float(torque_ref),
            "predicted_torque": float(predicted.torque_nm),
            "applied_torque": float(applied.torque_nm),
            "v_alpha_ref": float(v_alpha_ref),
            "v_beta_ref": float(v_beta_ref),
            "flux_abs": float(flux_abs),
            "loss_w": applied_loss_w,
            "switch_events": applied_switch_events,
            "accepted": 1.0 if decision.accepted else 0.0,
            "fault_flags": float(int(decision.fault_flags)),
            "tj_c": float(self.tj_c),
        }
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
    "SensorlessAdaptiveFocBaselineConfig",
    "SensorlessAdaptiveFocBaselineController",
]
