from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_safe_neural_horizon_pwm_novelty import COMPARISON_CONTROLLERS
from tools.run_safe_neural_horizon_pwm_study import BASE_CONTROLLER_SPECS, _make_base_params, _summarize_rows, run_trial


DEFAULT_STRESS_SCENARIOS = ["load_step", "overload", "dc_sag", "sensor_delay", "shock_load", "ood"]


def _metric(row: Dict[str, Any], name: str, field: str = "worst") -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get(field, 0.0))
    return float(value or 0.0)


def build_baseline_stress(
    *,
    mc: int = 3,
    steps: int = 80,
    seed: int = 23,
    scenarios: list[str] | None = None,
) -> Dict[str, Any]:
    scenario_names = list(scenarios or DEFAULT_STRESS_SCENARIOS)
    base_motor, inverter = _make_base_params()
    specs = [spec for spec in BASE_CONTROLLER_SPECS if spec[0] in COMPARISON_CONTROLLERS]
    rng = Random(seed)
    matrix: Dict[str, Any] = {}
    controllers: Dict[str, Any] = {}

    for scenario in scenario_names:
        scenario_payload: Dict[str, Any] = {}
        for label, horizon, feedback_period in specs:
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
        matrix[scenario] = scenario_payload

    for label, _, _ in specs:
        safety_worst = 0.0
        failure_count = 0
        finite_metrics = True
        max_speed_mean = 0.0
        max_current_mean = 0.0
        max_switch_mean = 0.0
        for scenario in scenario_names:
            row = dict(dict(matrix.get(scenario, {})).get(label, {}))
            try:
                safety_worst = max(safety_worst, _metric(row, "safety_violations", "worst"))
                max_speed_mean = max(max_speed_mean, _metric(row, "mean_abs_speed_error", "mean"))
                max_current_mean = max(max_current_mean, _metric(row, "mean_current_abs", "mean"))
                max_switch_mean = max(max_switch_mean, _metric(row, "switch_events", "mean"))
            except Exception:
                finite_metrics = False
            try:
                failure_count += int(row.get("failure_count", 0))
            except Exception:
                failure_count += 1
        controllers[label] = {
            "scenario_count": len(scenario_names),
            "safety_violations_worst": safety_worst,
            "unexpected_failure_count": failure_count,
            "finite_metrics": finite_metrics,
            "max_mean_abs_speed_error": max_speed_mean,
            "max_mean_current_abs": max_current_mean,
            "max_mean_switch_events": max_switch_mean,
            "stress_ready": safety_worst == 0.0 and failure_count == 0 and finite_metrics,
        }

    baseline_stress_ready = bool(controllers) and all(bool(row["stress_ready"]) for row in controllers.values())
    return {
        "status": "safe_neural_horizon_pwm_baseline_stress_evidence",
        "hardware_claim": False,
        "mc_trials": int(mc),
        "steps_per_trial": int(steps),
        "seed": int(seed),
        "scenarios": scenario_names,
        "controllers": controllers,
        "matrix": matrix,
        "baseline_stress_ready": baseline_stress_ready,
        "publication_tuning_claim": False,
        "interpretation": (
            "Bounded host stress evidence for separate comparison baselines. This is not a final "
            "publication tuning sweep and does not prove superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bounded stress evidence for SNH-PWM comparison baselines.")
    parser.add_argument("--mc", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario list.")
    parser.add_argument("--out-json", default=".tmp_pytest/safe_neural_horizon_pwm_baseline_stress.json")
    args = parser.parse_args()

    scenarios = [item.strip() for item in str(args.scenarios).split(",") if item.strip()] or None
    payload = build_baseline_stress(mc=args.mc, steps=args.steps, seed=args.seed, scenarios=scenarios)
    out = Path(args.out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(f"baseline_stress_ready: {payload['baseline_stress_ready']}")


if __name__ == "__main__":
    main()
