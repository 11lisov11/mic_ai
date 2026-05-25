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
class DtcHysteresisBaselineConfig:
    dt_s: float = 100.0e-6
    speed_kp: float = 0.050
    speed_ki: float = 0.90
    torque_limit_nm: float = 2.8
    flux_ref_wb: float | None = None
    torque_band_nm: float = 0.06
    flux_band_wb: float = 0.012
    torque_direction_weight: float = 2.6
    flux_direction_weight: float = 2.2
    hold_weight: float = 0.7
    current_weight: float = 0.55
    switching_weight: float = 0.075
    loss_weight: float = 0.0015
    common_mode_weight: float = 1.0e-4
    zero_vector_when_hold: bool = True
    zero_vector_active_demand_penalty: float = 0.60
    temp_trip_c: float = 125.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _hysteresis(error: float, band: float) -> int:
    if float(error) > float(band):
        return 1
    if float(error) < -float(band):
        return -1
    return 0


class DtcHysteresisBaselineController:
    """Host DTC hysteresis baseline with legal-vector and Safety Gateway output.

    This is not SVM and not neural shaping. It implements the DTC idea used for
    comparison: torque and stator-flux hysteresis comparators request increase,
    decrease, or hold; the baseline then chooses the legal inverter vector whose
    one-step predicted torque/flux movement best matches those requests.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: DtcHysteresisBaselineConfig | None = None,
    ) -> None:
        self.motor_params = motor_params
        self.inverter_params = inverter_params
        self.gateway = gateway
        self.cfg = cfg if cfg is not None else DtcHysteresisBaselineConfig(dt_s=inverter_params.t_pwm_s)
        self.twin = NeuralTwin(motor_params)
        self.speed_integral = 0.0
        self.last_torque = 0.0
        self.tj_c = float(inverter_params.ambient_c)

    def reset(self, state: AlphaBetaMotorState | None = None) -> None:
        self.gateway.reset(0)
        self.twin = NeuralTwin(self.motor_params, state)
        self.speed_integral = 0.0
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

    def _score_vector(
        self,
        *,
        vector_id: int,
        state: AlphaBetaMotorState,
        torque_now: float,
        flux_now: float,
        torque_ref: float,
        flux_ref: float,
        torque_cmd: int,
        flux_cmd: int,
        load_torque_nm: float,
        inverter: TwoLevelInverterParams,
        currents_alpha_beta: Tuple[float, float],
    ) -> tuple[float, Dict[str, float], AlphaBetaMotorState, float]:
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        v_alpha, v_beta = alpha_beta_voltage(vector_id, inverter, i_alpha_beta=currents_alpha_beta)
        step = model.next_state(v_alpha, v_beta, load_torque_nm, self.cfg.dt_s, state=state)
        flux_next = math.hypot(step.state.psi_s_alpha, step.state.psi_s_beta)
        torque_delta = step.torque_nm - torque_now
        flux_delta = flux_next - flux_now
        losses = estimate_inverter_losses(
            prev_vector_id=self.gateway.current_vector_id,
            next_vector_id=vector_id,
            params=inverter,
            i_alpha_beta=currents_alpha_beta,
        )

        if torque_cmd:
            torque_cost = max(0.0, -float(torque_cmd) * torque_delta) + 0.20 * abs(torque_ref - step.torque_nm)
        else:
            torque_cost = float(self.cfg.hold_weight) * abs(torque_delta)

        if flux_cmd:
            flux_cost = max(0.0, -float(flux_cmd) * flux_delta) + 0.35 * abs(flux_ref - flux_next)
        else:
            flux_cost = float(self.cfg.hold_weight) * abs(flux_delta)

        if self.cfg.zero_vector_when_hold and torque_cmd == 0 and flux_cmd == 0 and vector_id in (0, 7):
            torque_cost *= 0.75
            flux_cost *= 0.75
        if vector_id in (0, 7) and (torque_cmd > 0 or flux_cmd > 0):
            torque_cost += float(self.cfg.zero_vector_active_demand_penalty)
            flux_cost += float(self.cfg.zero_vector_active_demand_penalty)

        current_ratio = step.currents.stator_abs / max(float(self.motor_params.i_limit), 1e-9)
        current_cost = step.currents.stator_abs + 3.0 * max(0.0, current_ratio - 0.90) ** 2
        cost = (
            float(self.cfg.torque_direction_weight) * torque_cost
            + float(self.cfg.flux_direction_weight) * flux_cost
            + float(self.cfg.current_weight) * current_cost
            + float(self.cfg.switching_weight) * losses.switch_events
            + float(self.cfg.loss_weight) * losses.total_w
            + float(self.cfg.common_mode_weight) * abs(losses.common_mode_v)
        )
        metrics = {
            "candidate_cost": float(cost),
            "candidate_torque": float(step.torque_nm),
            "candidate_flux": float(flux_next),
            "candidate_current": float(step.currents.stator_abs),
            "candidate_torque_delta": float(torque_delta),
            "candidate_flux_delta": float(flux_delta),
            "candidate_switch_events": float(losses.switch_events),
            "candidate_loss_w": float(losses.total_w),
        }
        return float(cost), metrics, step.state, float(step.torque_nm)

    def _select_vector(
        self,
        *,
        state: AlphaBetaMotorState,
        torque_ref: float,
        load_torque_nm: float,
        inverter: TwoLevelInverterParams,
        currents_alpha_beta: Tuple[float, float],
    ) -> tuple[int, Dict[str, float]]:
        model = AlphaBetaInductionMotorModel(self.motor_params, state)
        currents = model.currents(state)
        torque_now = model.torque_nm(state, currents)
        flux_now = math.hypot(state.psi_s_alpha, state.psi_s_beta)
        flux_ref = self._flux_ref()
        torque_cmd = _hysteresis(torque_ref - torque_now, self.cfg.torque_band_nm)
        flux_cmd = _hysteresis(flux_ref - flux_now, self.cfg.flux_band_wb)

        best_vector = self.gateway.current_vector_id
        best_cost = float("inf")
        best_metrics: Dict[str, float] = {}
        for vector_id in range(8):
            cost, metrics, _, _ = self._score_vector(
                vector_id=vector_id,
                state=state,
                torque_now=torque_now,
                flux_now=flux_now,
                torque_ref=torque_ref,
                flux_ref=flux_ref,
                torque_cmd=torque_cmd,
                flux_cmd=flux_cmd,
                load_torque_nm=load_torque_nm,
                inverter=inverter,
                currents_alpha_beta=currents_alpha_beta,
            )
            if cost < best_cost:
                best_cost = cost
                best_vector = int(vector_id)
                best_metrics = metrics
        best_metrics.update(
            {
                "cost": float(best_cost),
                "torque_ref": float(torque_ref),
                "torque_now": float(torque_now),
                "flux_ref": float(flux_ref),
                "flux_now": float(flux_now),
                "torque_hysteresis_cmd": float(torque_cmd),
                "flux_hysteresis_cmd": float(flux_cmd),
            }
        )
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
        vector_id, metrics = self._select_vector(
            state=state,
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
        predicted_risk = max(0.0, predicted_i_abs / max(self.motor_params.i_limit, 1e-9) - 0.90)
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
    "DtcHysteresisBaselineConfig",
    "DtcHysteresisBaselineController",
]
