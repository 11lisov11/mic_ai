from __future__ import annotations

import argparse
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


def _controller(
    *,
    label: str,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    horizon: int,
    feedback_period: int,
) -> SafeNeuralHorizonPwmController:
    cfg = NeuralHorizonConfig(
        horizon=horizon,
        max_branching=4 if horizon >= 3 else 5,
        dt_s=inverter.t_pwm_s,
        feedback_base_period_steps=feedback_period,
        speed_kp=0.04 if "fcs" not in label else 0.035,
        switching_weight=0.04 if "protected" in label else 0.025,
        thermal_weight=0.02 if "thermal" in label else 0.01,
        feedback_weight=0.08 if "safe_neural" in label else 0.03,
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
        confidence_min=0.25,
        risk_max=1.4,
    )
    return SafeNeuralHorizonPwmController(base_motor, inverter, AIPwmSafetyGateway(limits), cfg)


def run_trial(
    *,
    label: str,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    rng: Random,
    steps: int,
    horizon: int,
    feedback_period: int,
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
    prev_vector = 0

    for k in range(max(int(steps), 1)):
        if k < steps // 5:
            omega_ref = 0.6 * omega_nom * (k / max(steps // 5, 1))
        else:
            omega_ref = 0.6 * omega_nom
        load_torque = 0.0 if k < steps // 2 else 0.35

        real_currents = real_motor.currents()
        measured_i_abs = real_currents.stator_abs
        speed_error_pre = omega_ref - controller.twin.state_hat.omega_m
        use_feedback = (
            k == 0
            or k % max(feedback_period, 1) == 0
            or abs(speed_error_pre) > controller.cfg.feedback_error_threshold_rad_s
            or controller.twin.uncertainty > controller.cfg.feedback_uncertainty_threshold
        )
        if use_feedback:
            feedback_count += 1

        result = controller.step(
            omega_ref=omega_ref,
            load_torque_nm=load_torque,
            measured_state=real_motor.state if use_feedback else None,
            measured_i_abs=measured_i_abs,
            vdc=inverter.Vdc,
        )
        if not result.decision.accepted:
            fallback_count += 1
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
                inverter,
                i_alpha_beta=(real_currents.i_s_alpha, real_currents.i_s_beta),
            )
        else:
            v_alpha, v_beta = 0.0, 0.0
        step = real_motor.step(v_alpha, v_beta, load_torque, inverter.t_pwm_s)
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
        "fault_latch_count": float(fault_latch_count),
        "safety_violations": float(safety_violations),
    }


def run_study(*, mc: int, steps: int, seed: int, quick: bool = False) -> Dict[str, object]:
    base_motor, inverter = _make_base_params()
    controller_specs = [
        ("protected_ai_pwm_h1_proxy", 1, 5),
        ("fcs_mpc_one_step_proxy", 1, 1),
        ("safe_neural_horizon_pwm_h2", 2, 10),
    ]
    if not quick:
        controller_specs.append(("safe_neural_horizon_pwm_h3_thermal", 3, 12))
        controller_specs.append(("safe_neural_horizon_pwm_h4_sparse", 4, 15))

    rng = Random(seed)
    out: Dict[str, object] = {
        "study": "Safe Neural Horizon PWM",
        "status": "host_simulation_only",
        "hardware_claim": False,
        "mc_trials": int(mc),
        "steps_per_trial": int(steps),
        "seed": int(seed),
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
            )
            for _ in range(max(int(mc), 1))
        ]
        metrics: Dict[str, object] = {}
        for key in rows[0].keys():
            metrics[key] = _summary([row[key] for row in rows])
        metrics["failure_count"] = int(
            sum(1 for row in rows if row["safety_violations"] > 0.0 or row["fault_latch_count"] > 0.0)
        )
        out["controllers"][label] = metrics
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Safe Neural Horizon PWM host-level research smoke/MC study.")
    parser.add_argument("--mc", type=int, default=8)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-json", default=".tmp_pytest/safe_neural_horizon_pwm_study.json")
    args = parser.parse_args()

    payload = run_study(mc=args.mc, steps=args.steps, seed=args.seed, quick=bool(args.quick))
    out = Path(args.out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(f"controllers: {len(payload['controllers'])}")


if __name__ == "__main__":
    main()
