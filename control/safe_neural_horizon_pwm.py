from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import math
from typing import Dict, Iterable, Sequence, Tuple

from models.induction_motor_alpha_beta import (
    AlphaBetaInductionMotorModel,
    AlphaBetaMotorParams,
    AlphaBetaMotorState,
)
from models.two_level_inverter import (
    TwoLevelInverterParams,
    alpha_beta_voltage,
    estimate_inverter_losses,
    switch_events,
)
from safety.ai_pwm_gateway import AIPwmRequest, AIPwmSafetyGateway, GateDecision


@dataclass(frozen=True)
class NeuralHorizonConfig:
    horizon: int = 2
    max_branching: int = 5
    dt_s: float = 100.0e-6
    speed_kp: float = 0.04
    speed_ki: float = 1.5
    torque_limit_nm: float = 4.0
    speed_weight: float = 1.0
    torque_weight: float = 0.7
    current_weight: float = 0.08
    switching_weight: float = 0.025
    loss_weight: float = 0.002
    thermal_weight: float = 0.01
    torque_ripple_weight: float = 0.05
    flux_weight: float = 0.6
    zero_vector_flux_penalty: float = 0.2
    flux_ref_wb: float | None = None
    feedback_weight: float = 0.04
    risk_weight: float = 0.4
    confidence_floor: float = 0.05
    feedback_base_period_steps: int = 10
    feedback_error_threshold_rad_s: float = 8.0
    feedback_uncertainty_threshold: float = 0.35
    temp_trip_c: float = 125.0


@dataclass(frozen=True)
class ControllerStepResult:
    decision: GateDecision
    vector_id: int
    feedback_requested: bool
    confidence: float
    predicted_i_abs: float
    predicted_risk: float
    metrics: Dict[str, float] = field(default_factory=dict)


class NeuralCostShaper:
    """Small deterministic MLP used as neural cost shaping, not a trained claim."""

    def shape(self, features: Sequence[float]) -> Dict[str, float]:
        x0, x1, x2, x3 = (float(v) for v in features[:4])
        h0 = math.tanh(1.1 * x0 + 0.4 * x1 - 0.2 * x2 + 0.1)
        h1 = math.tanh(-0.3 * x0 + 0.9 * x1 + 0.7 * x2 - 0.1)
        h2 = math.tanh(0.2 * x0 - 0.4 * x1 + 1.2 * x3)

        def scale(raw: float) -> float:
            return 0.65 + 0.7 / (1.0 + math.exp(-raw))

        return {
            "speed": scale(1.2 * h0 - 0.2 * h1),
            "current": scale(0.5 * h0 + 0.9 * h1),
            "switching": scale(-0.4 * h0 + 1.1 * h1 + 0.5 * h2),
            "thermal": scale(0.2 * h0 + 1.3 * h2),
            "risk": scale(0.4 * h1 + 1.0 * h2),
        }


class NeuralTwin:
    """Physics twin plus residual envelope and confidence estimate."""

    def __init__(self, params: AlphaBetaMotorParams, state: AlphaBetaMotorState | None = None) -> None:
        self.params = params
        self.state_hat = state if state is not None else AlphaBetaMotorState()
        self.residual_norm = 0.0
        self.uncertainty = 0.15

    def correct(self, measured: AlphaBetaMotorState, alpha: float = 0.35) -> None:
        alpha = max(0.0, min(1.0, float(alpha)))
        prev = self.state_hat
        residual = math.sqrt(
            (measured.psi_s_alpha - prev.psi_s_alpha) ** 2
            + (measured.psi_s_beta - prev.psi_s_beta) ** 2
            + 0.01 * (measured.omega_m - prev.omega_m) ** 2
        )
        self.residual_norm = 0.85 * self.residual_norm + 0.15 * residual
        self.uncertainty = max(0.02, 0.75 * self.uncertainty + 0.25 * min(1.0, residual))
        self.state_hat = replace(
            prev,
            psi_s_alpha=(1.0 - alpha) * prev.psi_s_alpha + alpha * measured.psi_s_alpha,
            psi_s_beta=(1.0 - alpha) * prev.psi_s_beta + alpha * measured.psi_s_beta,
            psi_r_alpha=(1.0 - alpha) * prev.psi_r_alpha + alpha * measured.psi_r_alpha,
            psi_r_beta=(1.0 - alpha) * prev.psi_r_beta + alpha * measured.psi_r_beta,
            omega_m=(1.0 - alpha) * prev.omega_m + alpha * measured.omega_m,
            theta_m=(1.0 - alpha) * prev.theta_m + alpha * measured.theta_m,
        )

    def predict(
        self,
        vector_id: int,
        inverter: TwoLevelInverterParams,
        load_torque_nm: float,
        dt_s: float,
    ) -> tuple[AlphaBetaMotorState, float, float]:
        model = AlphaBetaInductionMotorModel(self.params, self.state_hat)
        currents = model.currents()
        v_alpha, v_beta = alpha_beta_voltage(
            vector_id,
            inverter,
            i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
        )
        step = model.next_state(v_alpha, v_beta, load_torque_nm, dt_s)
        return step.state, step.torque_nm, step.currents.stator_abs

    def confidence(self) -> float:
        conf = math.exp(-self.uncertainty - 0.35 * self.residual_norm)
        return max(0.0, min(1.0, conf))

    def drift_without_feedback(self) -> None:
        self.uncertainty = min(1.0, self.uncertainty + 0.01)


class EventTriggeredFeedbackPolicy:
    def __init__(self, cfg: NeuralHorizonConfig) -> None:
        self.cfg = cfg
        self.step_count = 0
        self.feedback_count = 0

    @property
    def usage_ratio(self) -> float:
        return self.feedback_count / max(self.step_count, 1)

    def should_sample(self, *, speed_error: float, uncertainty: float, residual_norm: float) -> bool:
        self.step_count += 1
        periodic = self.step_count % max(int(self.cfg.feedback_base_period_steps), 1) == 0
        event = (
            abs(float(speed_error)) >= float(self.cfg.feedback_error_threshold_rad_s)
            or float(uncertainty) >= float(self.cfg.feedback_uncertainty_threshold)
            or float(residual_norm) >= 0.5
        )
        out = periodic or event
        if out:
            self.feedback_count += 1
        return out


class SafeNeuralHorizonPwmController:
    """Risk-aware neural horizon PWM controller protected by Safety Gateway."""

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: NeuralHorizonConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else NeuralHorizonConfig()
        self.twin = NeuralTwin(motor_params)
        self.shaper = NeuralCostShaper()
        self.feedback_policy = EventTriggeredFeedbackPolicy(self.cfg)
        self.speed_integral = 0.0
        self.last_torque = 0.0
        self.tj_c = float(inverter_params.ambient_c)
        self.theta_cmd = 0.0

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.feedback_policy = EventTriggeredFeedbackPolicy(self.cfg)
        self.speed_integral = 0.0
        self.last_torque = 0.0
        self.tj_c = float(self.inverter_params.ambient_c)
        self.theta_cmd = 0.0

    def _torque_ref(self, omega_ref: float, omega_hat: float) -> float:
        err = float(omega_ref) - float(omega_hat)
        self.speed_integral += err * float(self.cfg.dt_s)
        raw = float(self.cfg.speed_kp) * err + float(self.cfg.speed_ki) * self.speed_integral
        limit = max(float(self.cfg.torque_limit_nm), 1e-9)
        return max(-limit, min(limit, raw))

    def _flux_ref(self) -> float:
        if self.cfg.flux_ref_wb is not None and self.cfg.flux_ref_wb > 0.0:
            return float(self.cfg.flux_ref_wb)
        return max(0.02, min(0.45, 0.15 * float(self.motor_params.Lm) * max(self.motor_params.i_limit, 1e-3)))

    def _candidate_vectors(self, state: AlphaBetaMotorState, torque_ref: float, load_torque_nm: float) -> list[int]:
        scored: list[tuple[float, int]] = []
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        currents = model.currents()
        flux_ref = self._flux_ref()
        target_alpha = math.cos(self.theta_cmd)
        target_beta = math.sin(self.theta_cmd)
        for vector_id in range(8):
            v_alpha, v_beta = alpha_beta_voltage(
                vector_id,
                self.inverter_params,
                i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            )
            step = model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=state)
            flux_abs = math.hypot(step.state.psi_s_alpha, step.state.psi_s_beta)
            voltage_projection = v_alpha * target_alpha + v_beta * target_beta
            cost = abs(step.torque_nm - torque_ref) + 0.02 * step.currents.stator_abs
            cost += 0.8 * abs(flux_ref - flux_abs)
            if flux_abs < 0.8 * flux_ref:
                cost -= 0.003 * max(0.0, voltage_projection)
                if vector_id in (0, 7):
                    cost += float(self.cfg.zero_vector_flux_penalty)
            cost += 0.02 * switch_events(self.gateway.current_vector_id, vector_id)
            scored.append((cost, vector_id))
        scored.sort(key=lambda item: item[0])
        return [vector_id for _, vector_id in scored[: max(1, int(self.cfg.max_branching))]]

    def _score_sequence(
        self,
        sequence: Sequence[int],
        *,
        state: AlphaBetaMotorState,
        omega_ref: float,
        torque_ref: float,
        load_torque_nm: float,
        feedback_requested: bool,
    ) -> tuple[float, Dict[str, float]]:
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        local_state = state
        prev_vector = self.gateway.current_vector_id
        total_loss = 0.0
        total_switch = 0
        total_current = 0.0
        total_ripple = 0.0
        peak_i = 0.0
        torque_last = self.last_torque
        cmv_abs = 0.0
        flux_ref = self._flux_ref()
        flux_error_total = 0.0

        speed_norm = abs(float(omega_ref) - float(state.omega_m)) / max(abs(float(omega_ref)), 1.0)
        current_ratio = 0.0
        temp_ratio = max(0.0, (self.tj_c - self.inverter_params.ambient_c) / max(self.cfg.temp_trip_c, 1.0))
        shape = self.shaper.shape((speed_norm, current_ratio, self.twin.uncertainty, temp_ratio))

        for vector_id in sequence:
            currents = model.currents(local_state)
            v_alpha, v_beta = alpha_beta_voltage(
                vector_id,
                self.inverter_params,
                i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            )
            step = model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=local_state)
            losses = estimate_inverter_losses(
                prev_vector_id=prev_vector,
                next_vector_id=vector_id,
                params=self.inverter_params,
                i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
            )
            total_loss += losses.total_w
            total_switch += losses.switch_events
            total_current += step.currents.stator_abs
            peak_i = max(peak_i, step.currents.stator_abs)
            total_ripple += abs(step.torque_nm - torque_last)
            flux_abs = math.hypot(step.state.psi_s_alpha, step.state.psi_s_beta)
            flux_error_total += abs(flux_ref - flux_abs)
            if flux_abs < 0.8 * flux_ref and vector_id in (0, 7):
                flux_error_total += float(self.cfg.zero_vector_flux_penalty)
            cmv_abs += abs(losses.common_mode_v)
            torque_last = step.torque_nm
            local_state = step.state
            prev_vector = vector_id

        n = max(len(sequence), 1)
        speed_error = abs(float(omega_ref) - local_state.omega_m)
        torque_error = abs(torque_ref - torque_last)
        current_mean = total_current / n
        risk = self.twin.uncertainty
        if self.motor_params.i_limit > 0.0:
            risk += max(0.0, peak_i / self.motor_params.i_limit - 0.8)
        feedback_penalty = 1.0 if feedback_requested else 0.0

        cost = (
            self.cfg.speed_weight * shape["speed"] * speed_error
            + self.cfg.torque_weight * torque_error
            + self.cfg.current_weight * shape["current"] * current_mean
            + self.cfg.flux_weight * (flux_error_total / n)
            + self.cfg.switching_weight * shape["switching"] * total_switch
            + self.cfg.loss_weight * total_loss
            + self.cfg.torque_ripple_weight * total_ripple
            + self.cfg.feedback_weight * feedback_penalty
            + self.cfg.risk_weight * shape["risk"] * risk
            + self.cfg.thermal_weight * shape["thermal"] * max(0.0, self.tj_c - self.inverter_params.ambient_c)
            + 1.0e-4 * cmv_abs
        )
        return float(cost), {
            "speed_error": float(speed_error),
            "torque_error": float(torque_error),
            "current_mean": float(current_mean),
            "peak_i": float(peak_i),
            "switch_events": float(total_switch),
            "loss_w": float(total_loss),
            "torque_ripple_proxy": float(total_ripple),
            "flux_error": float(flux_error_total / n),
            "risk": float(risk),
            "feedback_usage_ratio": float(self.feedback_policy.usage_ratio),
        }

    def select_sequence(
        self,
        *,
        omega_ref: float,
        load_torque_nm: float,
        feedback_requested: bool,
    ) -> tuple[tuple[int, ...], Dict[str, float]]:
        state = self.twin.state_hat
        torque_ref = self._torque_ref(omega_ref, state.omega_m)
        candidates = self._candidate_vectors(state, torque_ref, load_torque_nm)
        horizon = max(1, min(int(self.cfg.horizon), 4))
        best_cost = float("inf")
        best_sequence: tuple[int, ...] = (self.gateway.current_vector_id,)
        best_metrics: Dict[str, float] = {}
        for sequence in product(candidates, repeat=horizon):
            cost, metrics = self._score_sequence(
                sequence,
                state=state,
                omega_ref=omega_ref,
                torque_ref=torque_ref,
                load_torque_nm=load_torque_nm,
                feedback_requested=feedback_requested,
            )
            if cost < best_cost:
                best_cost = cost
                best_sequence = tuple(int(v) for v in sequence)
                best_metrics = metrics
        best_metrics["cost"] = float(best_cost)
        best_metrics["torque_ref"] = float(torque_ref)
        return best_sequence, best_metrics

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
            self.twin.correct(measured_state)
        else:
            self.twin.drift_without_feedback()

        omega_e_cmd = float(self.motor_params.p) * float(omega_ref)
        self.theta_cmd = (self.theta_cmd + omega_e_cmd * float(self.cfg.dt_s)) % (2.0 * math.pi)

        speed_error = float(omega_ref) - float(self.twin.state_hat.omega_m)
        feedback_requested = self.feedback_policy.should_sample(
            speed_error=speed_error,
            uncertainty=self.twin.uncertainty,
            residual_norm=self.twin.residual_norm,
        )
        sequence, metrics = self.select_sequence(
            omega_ref=omega_ref,
            load_torque_nm=load_torque_nm,
            feedback_requested=feedback_requested,
        )

        vector_id = sequence[0]
        predicted_state, predicted_torque, predicted_i_abs = self.twin.predict(
            vector_id,
            self.inverter_params,
            load_torque_nm,
            self.cfg.dt_s,
        )
        confidence = max(float(self.cfg.confidence_floor), self.twin.confidence())
        predicted_risk = float(metrics.get("risk", self.twin.uncertainty))
        decision = self.gateway.evaluate(
            AIPwmRequest(
                vector_id=vector_id,
                dwell_s=self.cfg.dt_s,
                confidence=confidence,
                predicted_i_abs=predicted_i_abs,
                measured_i_abs=measured_i_abs,
                vdc=float(self.inverter_params.Vdc if vdc is None else vdc),
                tj_c=self.tj_c,
                predicted_risk=predicted_risk,
            )
        )

        applied_vector = decision.vector_id if decision.pwm_enabled else 0
        applied_state, applied_torque, _ = self.twin.predict(
            applied_vector,
            self.inverter_params,
            load_torque_nm,
            self.cfg.dt_s,
        )
        self.twin.state_hat = applied_state
        self.last_torque = applied_torque

        loss_w = float(metrics.get("loss_w", 0.0))
        tau = max(self.inverter_params.thermal_rth_k_per_w * self.inverter_params.thermal_cth_j_per_k, 1e-9)
        target_temp = self.inverter_params.ambient_c + self.inverter_params.thermal_rth_k_per_w * loss_w
        self.tj_c += (target_temp - self.tj_c) * min(1.0, self.cfg.dt_s / tau)

        metrics = dict(metrics)
        metrics.update(
            {
                "predicted_torque": float(predicted_torque),
                "accepted": 1.0 if decision.accepted else 0.0,
                "fault_flags": float(int(decision.fault_flags)),
                "tj_c": float(self.tj_c),
            }
        )
        return ControllerStepResult(
            decision=decision,
            vector_id=applied_vector,
            feedback_requested=feedback_requested,
            confidence=confidence,
            predicted_i_abs=float(predicted_i_abs),
            predicted_risk=predicted_risk,
            metrics=metrics,
        )


__all__ = [
    "ControllerStepResult",
    "EventTriggeredFeedbackPolicy",
    "NeuralCostShaper",
    "NeuralHorizonConfig",
    "NeuralTwin",
    "SafeNeuralHorizonPwmController",
]
