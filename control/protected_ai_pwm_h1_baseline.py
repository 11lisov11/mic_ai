from __future__ import annotations

from dataclasses import replace

from control.safe_neural_horizon_pwm import (
    ControllerStepResult,
    NeuralHorizonConfig,
    SafeNeuralHorizonPwmController,
)
from models.induction_motor_alpha_beta import AlphaBetaMotorParams, AlphaBetaMotorState
from models.two_level_inverter import TwoLevelInverterParams
from safety.ai_pwm_gateway import AIPwmSafetyGateway


def protected_h1_config(*, dt_s: float, feedback_period: int = 5) -> NeuralHorizonConfig:
    """Return the fixed prior protected AI-PWM H1 baseline configuration."""

    return NeuralHorizonConfig(
        horizon=1,
        max_branching=5,
        dt_s=float(dt_s),
        feedback_base_period_steps=max(int(feedback_period), 1),
        speed_kp=0.04,
        speed_ki=1.5,
        current_weight=0.08,
        switching_weight=0.04,
        thermal_weight=0.01,
        feedback_weight=0.04,
        flux_weight=0.6,
        torque_ripple_weight=0.05,
        risk_weight=0.55,
        feedback_error_threshold_rad_s=8.0,
        feedback_uncertainty_threshold=0.35,
    )


class ProtectedAiPwmH1BaselineController(SafeNeuralHorizonPwmController):
    """Prior protected AI-PWM H1 host baseline.

    This is the previous protected one-step AI-PWM architecture used for
    comparison with the new Safe Neural Horizon PWM H2/H3/H4 variants. The class
    fixes the old H=1 policy explicitly, instead of relying on a generic
    controller label that looked like a proxy in the release matrix.
    """

    def __init__(
        self,
        motor_params: AlphaBetaMotorParams,
        inverter_params: TwoLevelInverterParams,
        gateway: AIPwmSafetyGateway,
        cfg: NeuralHorizonConfig | None = None,
    ) -> None:
        fixed_cfg = cfg if cfg is not None else protected_h1_config(dt_s=inverter_params.t_pwm_s)
        fixed_cfg = replace(fixed_cfg, horizon=1, max_branching=max(1, int(fixed_cfg.max_branching)))
        super().__init__(motor_params, inverter_params, gateway, fixed_cfg)

    def step(
        self,
        *,
        omega_ref: float,
        load_torque_nm: float,
        measured_state: AlphaBetaMotorState | None = None,
        measured_i_abs: float = 0.0,
        vdc: float | None = None,
    ) -> ControllerStepResult:
        result = super().step(
            omega_ref=omega_ref,
            load_torque_nm=load_torque_nm,
            measured_state=measured_state,
            measured_i_abs=measured_i_abs,
            vdc=vdc,
        )
        metrics = dict(result.metrics)
        metrics["prior_protected_h1_baseline"] = 1.0
        metrics["horizon"] = 1.0
        return replace(result, metrics=metrics)


__all__ = [
    "ProtectedAiPwmH1BaselineController",
    "protected_h1_config",
]
