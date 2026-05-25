from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.deadbeat_current_baseline import DeadbeatCurrentBaselineConfig, DeadbeatCurrentBaselineController
from control.dtc_baseline import DtcHysteresisBaselineConfig, DtcHysteresisBaselineController
from control.dtc_svm_baseline import DtcSvmBaselineConfig, DtcSvmBaselineController
from control.fcs_mpc_baseline import FcsMpcOneStepBaselineConfig, FcsMpcOneStepBaselineController
from control.foc_svm_key_baseline import FocSvmKeyBaselineConfig, FocSvmKeyBaselineController
from control.protected_ai_pwm_h1_baseline import ProtectedAiPwmH1BaselineController, protected_h1_config
from control.sensorless_adaptive_foc_baseline import (
    SensorlessAdaptiveFocBaselineConfig,
    SensorlessAdaptiveFocBaselineController,
)
from models.induction_motor_alpha_beta import AlphaBetaInductionMotorModel, AlphaBetaMotorParams, AlphaBetaMotorState
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, switch_events
from safety.ai_pwm_gateway import AIPwmSafetyGateway, GatewayLimits, has_shoot_through, transition_waveform
from tools.check_safe_neural_horizon_pwm_novelty import COMPARISON_CONTROLLERS
from tools.run_safe_neural_horizon_pwm_study import (
    BASE_CONTROLLER_SPECS,
    _make_base_params,
    _scenario_values,
    _summarize_rows,
    randomized_motor_params,
)


DEFAULT_TUNING_SCENARIOS = ["load_step", "overload", "dc_sag", "ood"]


def _limits(base_motor: AlphaBetaMotorParams, inverter: TwoLevelInverterParams, *, confidence_min: float = 0.35) -> GatewayLimits:
    return GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=max(2.5 * base_motor.i_limit, 3.5),
        i_trip_a=max(3.5 * base_motor.i_limit, 5.0),
        vdc_min_v=0.4 * inverter.Vdc,
        vdc_max_v=1.25 * inverter.Vdc,
        tj_trip_c=125.0,
        confidence_min=confidence_min,
        risk_max=1.6,
    )


def _config_variants(label: str, dt_s: float) -> dict[str, Any]:
    if label == "protected_ai_pwm_h1_baseline":
        base = protected_h1_config(dt_s=dt_s, feedback_period=5)
        return {
            "default": base,
            "low_switching": replace(base, switching_weight=0.075, risk_weight=0.50),
            "current_guard": replace(base, current_weight=0.14, risk_weight=0.70),
            "tracking_bias": replace(base, speed_kp=0.048, speed_ki=1.8, switching_weight=0.035),
        }
    if label == "fcs_mpc_one_step_baseline":
        base = FcsMpcOneStepBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, current_weight=0.24, torque_limit_nm=3.2),
            "tracking_bias": replace(base, speed_kp=0.050, torque_weight=1.20, flux_weight=0.95),
            "low_switching": replace(base, switching_weight=0.060, current_weight=0.18),
        }
    if label == "foc_svm_key_baseline":
        base = FocSvmKeyBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, id_max_fraction=0.38, iq_max_fraction=0.70, switching_tiebreak_weight=0.025),
            "tracking_bias": replace(base, speed_kp=0.045, speed_ki=1.25, current_kp=26.0),
            "low_switching": replace(base, switching_tiebreak_weight=0.060, voltage_limit_fraction=0.86),
        }
    if label == "dtc_hysteresis_baseline":
        base = DtcHysteresisBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, current_weight=0.75, torque_limit_nm=2.5),
            "tracking_bias": replace(base, speed_kp=0.060, torque_direction_weight=3.0),
            "low_switching": replace(base, switching_weight=0.130, torque_band_nm=0.08, flux_band_wb=0.016),
        }
    if label == "dtc_svm_baseline":
        base = DtcSvmBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, current_weight=0.32, torque_limit_nm=2.7),
            "tracking_bias": replace(base, speed_kp=0.055, torque_voltage_kp=26.0),
            "low_switching": replace(base, switching_weight=0.070, voltage_limit_fraction=0.82),
        }
    if label == "deadbeat_current_baseline":
        base = DeadbeatCurrentBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, current_stress_weight=0.34, iq_max_fraction=0.66),
            "tracking_bias": replace(base, speed_kp=0.050, torque_weight=0.16),
            "low_switching": replace(base, switching_weight=1.45, voltage_error_weight=0.006),
        }
    if label == "sensorless_adaptive_foc_baseline":
        base = SensorlessAdaptiveFocBaselineConfig(dt_s=dt_s)
        return {
            "default": base,
            "current_guard": replace(base, id_max_fraction=0.36, iq_max_fraction=0.62, switching_tiebreak_weight=0.055),
            "tracking_bias": replace(base, speed_kp=0.040, observer_gain=0.24),
            "low_switching": replace(base, switching_tiebreak_weight=0.080, voltage_limit_fraction=0.80),
        }
    raise KeyError(label)


def _controller(
    label: str,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    cfg: Any,
) -> Any:
    gateway = AIPwmSafetyGateway(_limits(base_motor, inverter, confidence_min=0.25 if label == "protected_ai_pwm_h1_baseline" else 0.35))
    if label == "protected_ai_pwm_h1_baseline":
        return ProtectedAiPwmH1BaselineController(base_motor, inverter, gateway, cfg)
    if label == "fcs_mpc_one_step_baseline":
        return FcsMpcOneStepBaselineController(base_motor, inverter, gateway, cfg)
    if label == "foc_svm_key_baseline":
        return FocSvmKeyBaselineController(base_motor, inverter, gateway, cfg)
    if label == "dtc_hysteresis_baseline":
        return DtcHysteresisBaselineController(base_motor, inverter, gateway, cfg)
    if label == "dtc_svm_baseline":
        return DtcSvmBaselineController(base_motor, inverter, gateway, cfg)
    if label == "deadbeat_current_baseline":
        return DeadbeatCurrentBaselineController(base_motor, inverter, gateway, cfg)
    if label == "sensorless_adaptive_foc_baseline":
        return SensorlessAdaptiveFocBaselineController(base_motor, inverter, gateway, cfg)
    raise KeyError(label)


def _run_trial_with_controller(
    *,
    label: str,
    cfg: Any,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    rng: Random,
    steps: int,
    feedback_period: int,
    scenario: str,
) -> Dict[str, float]:
    real_params = randomized_motor_params(base_motor, rng)
    scenario_name = str(scenario).strip().lower()
    if scenario_name == "ood":
        real_params = replace(
            real_params,
            Rs=real_params.Rs * 2.0,
            Rr=real_params.Rr * 0.45,
            Lm=max(1e-9, real_params.Lm * 0.55),
            J=max(1e-9, real_params.J * 2.5),
        )
    real_motor = AlphaBetaInductionMotorModel(real_params, AlphaBetaMotorState())
    controller = _controller(label, base_motor, inverter, cfg)
    controller.reset(AlphaBetaMotorState())

    omega_nom = 2.0 * math.pi * 50.0 / max(base_motor.p, 1)
    speed_errors: list[float] = []
    currents: list[float] = []
    torque_values: list[float] = []
    switch_total = 0
    fallback_count = 0
    fault_latch_count = 0
    safety_violations = 0
    feedback_count = 0
    rejected_count = 0
    prev_vector = 0
    measured_state_history: list[AlphaBetaMotorState] = []

    for k in range(max(int(steps), 1)):
        omega_ref, load_torque, vdc_scale, force_sensor_dropout = _scenario_values(scenario_name, k, steps, omega_nom)
        step_inverter = replace(inverter, Vdc=float(inverter.Vdc) * float(vdc_scale))
        real_currents = real_motor.currents()
        measured_i_abs = real_currents.stator_abs
        measured_state_history.append(real_motor.state)
        measured_state = real_motor.state
        if scenario_name == "sensor_delay" and len(measured_state_history) > 6:
            measured_state = measured_state_history[-6]

        speed_error_pre = omega_ref - controller.twin.state_hat.omega_m
        feedback_error_threshold = float(getattr(controller.cfg, "feedback_error_threshold_rad_s", 6.0))
        feedback_uncertainty_threshold = float(getattr(controller.cfg, "feedback_uncertainty_threshold", 0.40))
        use_feedback = (
            k == 0
            or k % max(int(feedback_period), 1) == 0
            or abs(speed_error_pre) > feedback_error_threshold
            or controller.twin.uncertainty > feedback_uncertainty_threshold
        )
        if force_sensor_dropout and k > steps // 4:
            use_feedback = k % max(int(feedback_period) * 6, 1) == 0
        if use_feedback:
            feedback_count += 1

        result = controller.step(
            omega_ref=omega_ref,
            load_torque_nm=load_torque,
            measured_state=measured_state if use_feedback else None,
            measured_i_abs=measured_i_abs,
            vdc=step_inverter.Vdc,
        )
        if not result.decision.accepted:
            fallback_count += 1
            rejected_count += 1
        if result.decision.fault_latched:
            fault_latch_count += 1

        waveform = transition_waveform(prev_vector, result.vector_id, dead_time_ticks=2)
        if has_shoot_through(waveform):
            safety_violations += 1
        switch_total += switch_events(prev_vector, result.vector_id)
        prev_vector = result.vector_id

        if result.decision.pwm_enabled:
            v_alpha, v_beta = alpha_beta_voltage(
                result.vector_id,
                step_inverter,
                i_alpha_beta=(real_currents.i_s_alpha, real_currents.i_s_beta),
            )
        else:
            v_alpha, v_beta = 0.0, 0.0
        step = real_motor.step(v_alpha, v_beta, load_torque, step_inverter.t_pwm_s)
        speed_errors.append(abs(omega_ref - step.state.omega_m))
        currents.append(step.currents.stator_abs)
        torque_values.append(step.torque_nm)

    torque_ripple = 0.0
    if len(torque_values) > 1:
        torque_ripple = sum(abs(b - a) for a, b in zip(torque_values, torque_values[1:])) / (len(torque_values) - 1)
    return {
        "mean_abs_speed_error": sum(speed_errors) / max(len(speed_errors), 1),
        "p95_abs_speed_error": _percentile(speed_errors, 0.95),
        "mean_current_abs": sum(currents) / max(len(currents), 1),
        "max_current_abs": max(currents) if currents else 0.0,
        "torque_ripple_proxy": torque_ripple,
        "switch_events": float(switch_total),
        "feedback_usage_ratio": feedback_count / max(steps, 1),
        "fallback_count": float(fallback_count),
        "rejected_action_count": float(rejected_count),
        "fault_latch_count": float(fault_latch_count),
        "safety_violations": float(safety_violations),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = max(0.0, min(1.0, q)) * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _metric(row: Dict[str, Any], name: str, field: str = "mean") -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get(field, 0.0))
    return float(value or 0.0)


def _score(summary: Dict[str, Any]) -> float:
    return (
        _metric(summary, "mean_abs_speed_error")
        + 10.0 * _metric(summary, "mean_current_abs")
        + 0.035 * _metric(summary, "switch_events")
        + 2.0 * _metric(summary, "fallback_count")
        + 4.0 * _metric(summary, "torque_ripple_proxy")
        + 1000.0 * _metric(summary, "safety_violations", "worst")
        + 1000.0 * float(summary.get("failure_count", 0))
    )


def build_baseline_tuning(
    *,
    mc: int = 2,
    steps: int = 60,
    seed: int = 31,
    scenarios: list[str] | None = None,
) -> Dict[str, Any]:
    scenario_names = list(scenarios or DEFAULT_TUNING_SCENARIOS)
    base_motor, inverter = _make_base_params()
    spec_map = {label: (horizon, feedback_period) for label, horizon, feedback_period in BASE_CONTROLLER_SPECS}
    rng = Random(seed)
    controllers: Dict[str, Any] = {}

    for label in sorted(COMPARISON_CONTROLLERS):
        if label not in spec_map:
            continue
        _, feedback_period = spec_map[label]
        variants: Dict[str, Any] = {}
        for variant_name, cfg in _config_variants(label, inverter.t_pwm_s).items():
            scenario_summaries: Dict[str, Any] = {}
            all_rows: list[Dict[str, float]] = []
            for scenario in scenario_names:
                rows = [
                    _run_trial_with_controller(
                        label=label,
                        cfg=cfg,
                        base_motor=base_motor,
                        inverter=inverter,
                        rng=rng,
                        steps=steps,
                        feedback_period=feedback_period,
                        scenario=scenario,
                    )
                    for _ in range(max(int(mc), 1))
                ]
                summary = _summarize_rows(rows)
                scenario_summaries[scenario] = summary
                all_rows.extend(rows)
            aggregate = _summarize_rows(all_rows)
            variants[variant_name] = {
                "score": _score(aggregate),
                "aggregate": aggregate,
                "scenarios": scenario_summaries,
                "safety_violations_worst": _metric(aggregate, "safety_violations", "worst"),
                "unexpected_failure_count": int(aggregate.get("failure_count", 0)),
            }
        selected_name = min(variants, key=lambda name: float(variants[name]["score"]))
        default_score = float(variants["default"]["score"])
        selected_score = float(variants[selected_name]["score"])
        controllers[label] = {
            "candidate_count": len(variants),
            "selected_variant": selected_name,
            "default_score": default_score,
            "selected_score": selected_score,
            "improvement_vs_default_pct": 100.0 * (default_score - selected_score) / max(default_score, 1e-9),
            "safety_violations_worst": float(variants[selected_name]["safety_violations_worst"]),
            "unexpected_failure_count": int(variants[selected_name]["unexpected_failure_count"]),
            "tuning_ready": selected_score <= default_score + 1e-9
            and float(variants[selected_name]["safety_violations_worst"]) == 0.0
            and int(variants[selected_name]["unexpected_failure_count"]) == 0
            and len(variants) >= 3,
            "variants": variants,
        }

    baseline_tuning_ready = bool(controllers) and all(bool(row["tuning_ready"]) for row in controllers.values())
    return {
        "status": "safe_neural_horizon_pwm_baseline_tuning_evidence",
        "hardware_claim": False,
        "mc_trials": int(mc),
        "steps_per_trial": int(steps),
        "seed": int(seed),
        "scenarios": scenario_names,
        "baseline_tuning_ready": baseline_tuning_ready,
        "publication_tuning_claim": baseline_tuning_ready,
        "superiority_claim": False,
        "controllers": controllers,
        "score_definition": {
            "mean_abs_speed_error": 1.0,
            "mean_current_abs": 10.0,
            "switch_events": 0.035,
            "fallback_count": 2.0,
            "torque_ripple_proxy": 4.0,
            "safety_violations_worst": 1000.0,
            "failure_count": 1000.0,
        },
        "interpretation": (
            "Bounded host parameter-sweep evidence for comparison baselines. It selects a safe baseline variant "
            "from a small fixed grid and supports fairer host comparison; it does not claim SNH-PWM superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded parameter-sweep tuning evidence for SNH-PWM baselines.")
    parser.add_argument("--mc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario list.")
    parser.add_argument("--out-json", default=".tmp_pytest/safe_neural_horizon_pwm_baseline_tuning.json")
    args = parser.parse_args()

    scenarios = [item.strip() for item in str(args.scenarios).split(",") if item.strip()] or None
    payload = build_baseline_tuning(mc=args.mc, steps=args.steps, seed=args.seed, scenarios=scenarios)
    out = Path(args.out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(f"baseline_tuning_ready: {payload['baseline_tuning_ready']}")


if __name__ == "__main__":
    main()
