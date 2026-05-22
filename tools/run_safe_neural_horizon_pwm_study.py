from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import create_default_env
from control.safe_neural_horizon_pwm import NeuralHorizonConfig, SafeNeuralHorizonPwmController
from models.induction_motor_alpha_beta import (
    AlphaBetaInductionMotorModel,
    AlphaBetaMotorParams,
    AlphaBetaMotorState,
    randomized_motor_params,
)
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, switch_events
from safety.ai_pwm_gateway import AIPwmSafetyGateway, GatewayLimits, has_shoot_through, transition_waveform


BASE_CONTROLLER_SPECS = [
    ("protected_ai_pwm_h1_proxy", 1, 5),
    ("fcs_mpc_one_step_proxy", 1, 1),
    ("foc_svm_key_proxy", 1, 1),
    ("dtc_hysteresis_proxy", 1, 1),
    ("dtc_svm_proxy", 1, 1),
    ("deadbeat_current_proxy", 1, 1),
    ("sensorless_adaptive_foc_proxy", 1, 8),
    ("safe_neural_horizon_pwm_h2", 2, 10),
]

EXTENDED_CONTROLLER_SPECS = [
    ("safe_neural_horizon_pwm_h3_thermal", 3, 12),
    ("safe_neural_horizon_pwm_h4_sparse", 4, 15),
]

DEFAULT_SCENARIOS = [
    "start_no_load",
    "start_with_load",
    "load_step",
    "load_shed",
    "reverse",
    "low_speed",
    "dc_sag",
    "sensor_dropout",
]

ABLATION_SPECS = [
    ("ablation_h1_no_horizon", 1, 10),
    ("ablation_h2_dense_feedback", 2, 1),
    ("ablation_h2_sparse_feedback", 2, 25),
    ("ablation_h2_low_switching", 2, 10),
    ("ablation_h2_low_current", 2, 10),
]


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    pos = max(0.0, min(1.0, q)) * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _summary(values: Iterable[float]) -> Dict[str, float]:
    arr = [float(v) for v in values]
    if not arr:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "worst": 0.0}
    return {
        "mean": sum(arr) / len(arr),
        "median": _percentile(arr, 0.5),
        "p05": _percentile(arr, 0.05),
        "p95": _percentile(arr, 0.95),
        "worst": max(arr),
    }


def _make_base_params() -> tuple[AlphaBetaMotorParams, TwoLevelInverterParams]:
    env = create_default_env()
    motor = AlphaBetaMotorParams.from_motor_params(env.motor)
    inverter = TwoLevelInverterParams(
        Vdc=float(env.inverter.Vdc),
        f_pwm=float(env.inverter.f_pwm),
        dead_time_s=1.0e-6,
        min_pulse_s=2.0e-6,
        r_on_ohm=0.08,
        v_drop_v=0.8,
        e_sw_j_per_a=2.0e-6,
    )
    return motor, inverter


def _controller_specs(quick: bool = False) -> list[tuple[str, int, int]]:
    specs = list(BASE_CONTROLLER_SPECS)
    if quick:
        return [
            ("protected_ai_pwm_h1_proxy", 1, 5),
            ("fcs_mpc_one_step_proxy", 1, 1),
            ("foc_svm_key_proxy", 1, 1),
            ("safe_neural_horizon_pwm_h2", 2, 10),
        ]
    specs.extend(EXTENDED_CONTROLLER_SPECS)
    return specs


def _controller(
    *,
    label: str,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    horizon: int,
    feedback_period: int,
) -> SafeNeuralHorizonPwmController:
    max_branching = 4 if horizon >= 3 else 5
    speed_kp = 0.04
    speed_ki = 1.5
    current_weight = 0.08
    switching_weight = 0.025
    thermal_weight = 0.01
    feedback_weight = 0.04
    flux_weight = 0.6
    torque_ripple_weight = 0.05
    risk_weight = 0.4
    feedback_error_threshold = 8.0
    confidence_min = 0.25

    if "fcs_mpc" in label:
        speed_kp = 0.035
        current_weight = 0.12
        switching_weight = 0.015
        feedback_weight = 0.0
    elif "foc_svm" in label:
        speed_kp = 0.03
        current_weight = 0.16
        switching_weight = 0.04
        torque_ripple_weight = 0.08
        flux_weight = 0.9
        feedback_weight = 0.0
        max_branching = 8
    elif "dtc_hysteresis" in label:
        speed_kp = 0.055
        current_weight = 0.05
        switching_weight = 0.008
        torque_ripple_weight = 0.16
        flux_weight = 0.75
    elif "dtc_svm" in label:
        speed_kp = 0.045
        current_weight = 0.08
        switching_weight = 0.03
        torque_ripple_weight = 0.12
        flux_weight = 0.8
    elif "deadbeat" in label:
        speed_kp = 0.05
        speed_ki = 0.8
        current_weight = 0.20
        switching_weight = 0.018
        torque_ripple_weight = 0.06
    elif "sensorless" in label:
        speed_kp = 0.028
        current_weight = 0.14
        switching_weight = 0.035
        feedback_weight = 0.12
        feedback_error_threshold = 40.0
        confidence_min = 0.15
    elif "protected" in label:
        switching_weight = 0.04
        risk_weight = 0.55

    if "thermal" in label:
        thermal_weight = 0.035
        switching_weight += 0.01
    if "sparse" in label:
        feedback_weight = 0.18
        feedback_error_threshold = 60.0
    if "dense_feedback" in label:
        feedback_period = 1
        feedback_weight = 0.0
    if "sparse_feedback" in label:
        feedback_period = max(feedback_period, 25)
        feedback_weight = 0.22
        feedback_error_threshold = 80.0
    if "low_switching" in label:
        switching_weight = 0.12
    if "low_current" in label:
        current_weight = 0.28

    cfg = NeuralHorizonConfig(
        horizon=horizon,
        max_branching=max_branching,
        dt_s=inverter.t_pwm_s,
        feedback_base_period_steps=feedback_period,
        speed_kp=speed_kp,
        speed_ki=speed_ki,
        current_weight=current_weight,
        switching_weight=switching_weight,
        thermal_weight=thermal_weight,
        feedback_weight=feedback_weight,
        flux_weight=flux_weight,
        torque_ripple_weight=torque_ripple_weight,
        risk_weight=risk_weight,
        feedback_error_threshold_rad_s=feedback_error_threshold,
    )
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=max(2.5 * base_motor.i_limit, 3.5),
        i_trip_a=max(3.5 * base_motor.i_limit, 5.0),
        vdc_min_v=0.4 * inverter.Vdc,
        vdc_max_v=1.25 * inverter.Vdc,
        tj_trip_c=125.0,
        confidence_min=confidence_min,
        risk_max=1.4,
    )
    return SafeNeuralHorizonPwmController(base_motor, inverter, AIPwmSafetyGateway(limits), cfg)


def _scenario_values(name: str, k: int, steps: int, omega_nom: float) -> tuple[float, float, float, bool]:
    """Return omega_ref, load_torque, vdc_scale, force_sensor_dropout."""

    name = str(name or "load_step").strip().lower()
    steps = max(int(steps), 1)
    ramp_steps = max(steps // 5, 1)
    progress = min(1.0, k / ramp_steps)

    if name == "start_no_load":
        return 0.6 * omega_nom * progress, 0.0, 1.0, False
    if name == "start_with_load":
        return 0.6 * omega_nom * progress, 0.35, 1.0, False
    if name == "load_shed":
        return 0.6 * omega_nom, 0.45 if k < steps // 2 else 0.0, 1.0, False
    if name == "reverse":
        ref = 0.45 * omega_nom if k < steps // 2 else -0.35 * omega_nom
        return ref, 0.25, 1.0, False
    if name == "low_speed":
        return 0.15 * omega_nom, 0.15, 1.0, False
    if name == "dc_sag":
        vdc_scale = 0.68 if steps // 3 <= k < (2 * steps) // 3 else 1.0
        return 0.55 * omega_nom, 0.3, vdc_scale, False
    if name == "sensor_dropout":
        return 0.55 * omega_nom * progress, 0.3, 1.0, True
    if name == "periodic_load":
        load = 0.25 + 0.15 * math.sin(2.0 * math.pi * k / max(steps // 4, 1))
        return 0.55 * omega_nom, load, 1.0, False
    # default: load step
    return 0.6 * omega_nom * progress if k < ramp_steps else 0.6 * omega_nom, 0.0 if k < steps // 2 else 0.35, 1.0, False


def run_trial(
    *,
    label: str,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    rng: Random,
    steps: int,
    horizon: int,
    feedback_period: int,
    scenario: str = "load_step",
) -> Dict[str, float]:
    real_params = randomized_motor_params(base_motor, rng)
    real_motor = AlphaBetaInductionMotorModel(real_params, AlphaBetaMotorState())
    controller = _controller(
        label=label,
        base_motor=base_motor,
        inverter=inverter,
        horizon=horizon,
        feedback_period=feedback_period,
    )
    controller.reset(AlphaBetaMotorState())

    omega_nom = 2.0 * math.pi * 50.0 / max(base_motor.p, 1)
    speed_errors: List[float] = []
    currents: List[float] = []
    torque_values: List[float] = []
    switch_total = 0
    fallback_count = 0
    fault_latch_count = 0
    safety_violations = 0
    feedback_count = 0
    rejected_count = 0
    undervoltage_steps = 0
    prev_vector = 0

    for k in range(max(int(steps), 1)):
        omega_ref, load_torque, vdc_scale, force_sensor_dropout = _scenario_values(scenario, k, steps, omega_nom)
        step_inverter = replace(inverter, Vdc=float(inverter.Vdc) * float(vdc_scale))
        if vdc_scale < 0.75:
            undervoltage_steps += 1

        real_currents = real_motor.currents()
        measured_i_abs = real_currents.stator_abs
        speed_error_pre = omega_ref - controller.twin.state_hat.omega_m
        use_feedback = (
            k == 0
            or k % max(feedback_period, 1) == 0
            or abs(speed_error_pre) > controller.cfg.feedback_error_threshold_rad_s
            or controller.twin.uncertainty > controller.cfg.feedback_uncertainty_threshold
        )
        if force_sensor_dropout and k > steps // 4:
            use_feedback = k % max(feedback_period * 6, 1) == 0
        if use_feedback:
            feedback_count += 1

        result = controller.step(
            omega_ref=omega_ref,
            load_torque_nm=load_torque,
            measured_state=real_motor.state if use_feedback else None,
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
        "undervoltage_steps": float(undervoltage_steps),
    }


def _summarize_rows(rows: list[Dict[str, float]]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for key in rows[0].keys():
        metrics[key] = _summary([row[key] for row in rows])
    metrics["failure_count"] = int(
        sum(1 for row in rows if row["safety_violations"] > 0.0 or row["fault_latch_count"] > 0.0)
    )
    return metrics


def _dominates(left: Dict[str, object], right: Dict[str, object], keys: list[str]) -> bool:
    left_vals = [float(dict(left[k]).get("mean", 0.0)) for k in keys]
    right_vals = [float(dict(right[k]).get("mean", 0.0)) for k in keys]
    return all(a <= b for a, b in zip(left_vals, right_vals)) and any(a < b for a, b in zip(left_vals, right_vals))


def pareto_front(controllers: Dict[str, object]) -> list[str]:
    keys = [
        "mean_abs_speed_error",
        "mean_current_abs",
        "torque_ripple_proxy",
        "switch_events",
        "feedback_usage_ratio",
        "fallback_count",
    ]
    labels = list(controllers.keys())
    front: list[str] = []
    for label in labels:
        current = dict(controllers[label])
        dominated = False
        for other_label in labels:
            if other_label == label:
                continue
            other = dict(controllers[other_label])
            if _dominates(other, current, keys):
                dominated = True
                break
        if not dominated:
            front.append(label)
    return front


def run_study(*, mc: int, steps: int, seed: int, quick: bool = False, scenario: str = "load_step") -> Dict[str, object]:
    base_motor, inverter = _make_base_params()
    controller_specs = _controller_specs(quick=quick)

    rng = Random(seed)
    out: Dict[str, object] = {
        "study": "Safe Neural Horizon PWM",
        "status": "host_simulation_only",
        "hardware_claim": False,
        "mc_trials": int(mc),
        "steps_per_trial": int(steps),
        "seed": int(seed),
        "scenario": str(scenario),
        "controllers": {},
    }
    for label, horizon, feedback_period in controller_specs:
        rows = [
            run_trial(
                label=label,
                base_motor=base_motor,
                inverter=inverter,
                rng=rng,
                steps=steps,
                horizon=horizon,
                feedback_period=feedback_period,
                scenario=scenario,
            )
            for _ in range(max(int(mc), 1))
        ]
        out["controllers"][label] = _summarize_rows(rows)
    out["pareto_front"] = pareto_front(dict(out["controllers"]))
    return out


def run_fault_injection_matrix() -> Dict[str, object]:
    limits = GatewayLimits(i_soft_a=3.0, i_trip_a=4.0, vdc_min_v=50.0, vdc_max_v=500.0)
    cases = {
        "invalid_vector": {"vector_id": 99, "dwell_s": 100e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 300.0, "tj_c": 40.0},
        "too_short_pulse": {"vector_id": 1, "dwell_s": 0.1e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 300.0, "tj_c": 40.0},
        "overcurrent": {"vector_id": 1, "dwell_s": 100e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 4.5, "vdc": 300.0, "tj_c": 40.0},
        "overtemperature": {"vector_id": 1, "dwell_s": 100e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 300.0, "tj_c": 130.0},
        "undervoltage": {"vector_id": 1, "dwell_s": 100e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 40.0, "tj_c": 40.0},
        "low_confidence": {"vector_id": 1, "dwell_s": 100e-6, "confidence": 0.1, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 300.0, "tj_c": 40.0},
        "watchdog": {"vector_id": 1, "dwell_s": 100e-6, "confidence": 0.9, "predicted_i_abs": 0.2, "measured_i_abs": 0.2, "vdc": 300.0, "tj_c": 40.0, "watchdog_ok": False},
    }
    results: Dict[str, object] = {}
    for name, payload in cases.items():
        gateway = AIPwmSafetyGateway(limits)
        req = payload.copy()
        watchdog_ok = bool(req.pop("watchdog_ok", True))
        from safety.ai_pwm_gateway import AIPwmRequest

        decision = gateway.evaluate(AIPwmRequest(**req, predicted_risk=0.1, watchdog_ok=watchdog_ok))
        results[name] = {
            "accepted": bool(decision.accepted),
            "pwm_enabled": bool(decision.pwm_enabled),
            "fault_flags": int(decision.fault_flags),
            "fault_latched": bool(decision.fault_latched),
            "shoot_through": bool(decision.gates.shoot_through),
        }
    return {
        "status": "host_gateway_fault_injection_only",
        "all_cases_no_shoot_through": all(not dict(row)["shoot_through"] for row in results.values()),
        "cases": results,
    }


def run_matrix(
    *,
    mc: int,
    steps: int,
    seed: int,
    quick: bool = False,
    scenarios: list[str] | None = None,
    include_ablation: bool = True,
) -> Dict[str, object]:
    scenario_names = scenarios if scenarios is not None else (DEFAULT_SCENARIOS[:3] if quick else DEFAULT_SCENARIOS)
    base_motor, inverter = _make_base_params()
    rng = Random(seed)
    controller_specs = _controller_specs(quick=quick)
    matrix: Dict[str, object] = {}
    for scenario in scenario_names:
        scenario_payload: Dict[str, object] = {}
        for label, horizon, feedback_period in controller_specs:
            rows = [
                run_trial(
                    label=label,
                    base_motor=base_motor,
                    inverter=inverter,
                    rng=rng,
                    steps=steps,
                    horizon=horizon,
                    feedback_period=feedback_period,
                    scenario=scenario,
                )
                for _ in range(max(int(mc), 1))
            ]
            scenario_payload[label] = _summarize_rows(rows)
        scenario_payload["pareto_front"] = pareto_front(
            {k: v for k, v in scenario_payload.items() if k != "pareto_front"}
        )
        matrix[scenario] = scenario_payload

    ablation: Dict[str, object] = {}
    if include_ablation:
        for label, horizon, feedback_period in ABLATION_SPECS:
            rows = [
                run_trial(
                    label=label,
                    base_motor=base_motor,
                    inverter=inverter,
                    rng=rng,
                    steps=steps,
                    horizon=horizon,
                    feedback_period=feedback_period,
                    scenario="load_step",
                )
                for _ in range(max(int(mc), 1))
            ]
            ablation[label] = _summarize_rows(rows)
        ablation["pareto_front"] = pareto_front({k: v for k, v in ablation.items() if k != "pareto_front"})

    return {
        "study": "Safe Neural Horizon PWM",
        "status": "host_simulation_matrix_only",
        "hardware_claim": False,
        "mc_trials": int(mc),
        "steps_per_trial": int(steps),
        "seed": int(seed),
        "scenarios": scenario_names,
        "matrix": matrix,
        "ablation": ablation,
        "fault_injection": run_fault_injection_matrix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Safe Neural Horizon PWM host-level research smoke/MC study.")
    parser.add_argument("--mc", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scenario", default="load_step")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario list for --matrix.")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--out-json", default=".tmp_pytest/safe_neural_horizon_pwm_study.json")
    args = parser.parse_args()

    scenario_list = [x.strip() for x in str(args.scenarios).split(",") if x.strip()] or None
    if bool(args.matrix):
        payload = run_matrix(
            mc=args.mc,
            steps=args.steps,
            seed=args.seed,
            quick=bool(args.quick),
            scenarios=scenario_list,
            include_ablation=not bool(args.no_ablation),
        )
    else:
        payload = run_study(
            mc=args.mc,
            steps=args.steps,
            seed=args.seed,
            quick=bool(args.quick),
            scenario=str(args.scenario),
        )
    out = Path(args.out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    if "controllers" in payload:
        print(f"controllers: {len(payload['controllers'])}")
    else:
        print(f"scenarios: {len(payload.get('scenarios', []))}")


if __name__ == "__main__":
    main()
