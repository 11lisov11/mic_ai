from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_safe_neural_horizon_pwm_study import DEFAULT_SCENARIOS


SAFE_CONTROLLER_VARIANTS = {
    "safe_neural_horizon_pwm_h2",
    "safe_neural_horizon_pwm_h3_thermal",
    "safe_neural_horizon_pwm_h4_sparse",
}
COMPARISON_CONTROLLERS = {
    "protected_ai_pwm_h1_proxy",
    "fcs_mpc_one_step_baseline",
    "foc_svm_key_baseline",
    "dtc_hysteresis_baseline",
    "dtc_svm_baseline",
    "deadbeat_current_baseline",
    "sensorless_adaptive_foc_baseline",
}
ABLATION_KEYS = {
    "ablation_h1_no_horizon",
    "ablation_h2_dense_feedback",
    "ablation_h2_sparse_feedback",
    "ablation_h2_low_switching",
    "ablation_h2_low_current",
    "pareto_front",
}


def _load_payload(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        path = path / "safe_neural_horizon_pwm_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_novelty(path: Path) -> Dict[str, Any]:
    payload = _load_payload(path)
    checks: Dict[str, Any] = {}
    failures: List[str] = []
    warnings: List[str] = []

    checks["host_simulation_status"] = str(payload.get("status", "")).startswith("host_")
    if not checks["host_simulation_status"]:
        failures.append("results status must be host-scoped")

    checks["hardware_claim_false"] = payload.get("hardware_claim") is False
    if not checks["hardware_claim_false"]:
        failures.append("hardware_claim must be false")

    scenarios = list(payload.get("scenarios", []))
    missing_scenarios = [name for name in DEFAULT_SCENARIOS if name not in scenarios]
    checks["full_tz_scenario_set_present"] = not missing_scenarios
    checks["scenario_count"] = len(scenarios)
    if missing_scenarios:
        failures.append(f"missing required scenarios: {missing_scenarios}")

    matrix = dict(payload.get("matrix", {}))
    controller_missing: Dict[str, list[str]] = {}
    required_controllers = SAFE_CONTROLLER_VARIANTS | COMPARISON_CONTROLLERS
    for scenario in scenarios:
        rows = dict(matrix.get(scenario, {}))
        missing = sorted(required_controllers - set(rows.keys()))
        if missing:
            controller_missing[scenario] = missing
    checks["safe_variants_present"] = not any(
        set(SAFE_CONTROLLER_VARIANTS) - set(dict(matrix.get(scenario, {})).keys()) for scenario in scenarios
    )
    checks["comparison_controllers_present"] = not controller_missing
    if controller_missing:
        failures.append(f"missing comparison controllers: {controller_missing}")

    ablation = dict(payload.get("ablation", {}))
    missing_ablation = sorted(ABLATION_KEYS - set(ablation.keys()))
    checks["ablation_smoke_present"] = not missing_ablation
    if missing_ablation:
        failures.append(f"missing ablation keys: {missing_ablation}")

    fault = dict(payload.get("fault_injection", {}))
    no_deadtime = dict(dict(fault.get("cases", {})).get("no_deadtime_transition_emulation", {}))
    checks["no_shoot_through_gateway_cases"] = bool(fault.get("all_gateway_cases_no_shoot_through", False))
    checks["raw_shoot_through_detector_triggered"] = bool(fault.get("raw_shoot_through_detector_triggered", False))
    checks["deadtime_path_detector_triggered"] = bool(
        no_deadtime.get("direct_leg_transition_without_deadtime", False)
        and no_deadtime.get("safe_deadtime_path_valid", False)
        and no_deadtime.get("blocked_by_gateway_deadtime_path", False)
    )
    if not checks["no_shoot_through_gateway_cases"]:
        failures.append("gateway fault injection must show no accepted shoot-through")
    if not checks["raw_shoot_through_detector_triggered"]:
        failures.append("raw shoot-through detector must trigger on illegal raw gate emulation")
    if not checks["deadtime_path_detector_triggered"]:
        failures.append("dead-time path detector must distinguish direct HIGH/LOW transitions")

    checks["novel_control_tuple_present"] = all(
        [
            checks["safe_variants_present"],
            checks["ablation_smoke_present"],
            checks["no_shoot_through_gateway_cases"],
            checks["deadtime_path_detector_triggered"],
        ]
    )

    mc_trials = int(payload.get("mc_trials", 0))
    checks["first_study_mc_minimum_met"] = mc_trials >= 3
    checks["publication_mc_minimum_met"] = mc_trials >= 500
    if not checks["publication_mc_minimum_met"]:
        warnings.append("publication-scale MC is not met; current tracked release is host evidence only")

    warnings.append("FOC-SVM, FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC are separate host baselines; they are not final publication-tuned")
    warnings.append("neural twin is a scaffold, not a trained domain-randomized ensemble")

    host_novelty_claim_supported = not failures
    return {
        "status": "safe_neural_horizon_pwm_novelty_audit",
        "host_novelty_claim_supported": host_novelty_claim_supported,
        "allowed_claim": (
            "SNH-PWM is a distinct host-simulated control architecture combining event-triggered twin feedback, "
            "neural cost shaping, horizon inverter-vector search, and protected AI-PWM Safety Gateway."
        ),
        "not_allowed_claims": [
            "publication-grade superiority over tuned FOC-SVM/FCS-MPC/DTC baselines",
            "MCU/HIL/bench readiness",
            "full no-feedback control",
            "trained neural-twin optimality",
        ],
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Safe Neural Horizon PWM novelty claim scope.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = analyze_novelty(Path(args.input).expanduser().resolve())
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out}")
    print(f"host_novelty_claim_supported: {result['host_novelty_claim_supported']}")
    if args.strict and not bool(result["host_novelty_claim_supported"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
