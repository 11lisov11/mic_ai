from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_safe_neural_horizon_pwm_novelty import (
    ABLATION_KEYS,
    COMPARISON_CONTROLLERS,
    SAFE_CONTROLLER_VARIANTS,
    analyze_novelty,
)
from tools.run_safe_neural_horizon_pwm_study import DEFAULT_SCENARIOS


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release(path: Path) -> tuple[Dict[str, Any], Path | None, Dict[str, Any] | None]:
    if path.is_dir():
        result_path = path / "safe_neural_horizon_pwm_results.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        mc100_path = path / "safe_neural_horizon_pwm_mc100_smoke.json"
        return _load_json(result_path), path, _load_json(mc100_path) if mc100_path.exists() else None
    return _load_json(path), None, None


def _source_contains(path: Path, names: list[str]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(name in text for name in names)


def _matrix_has_required_controllers(payload: Dict[str, Any]) -> bool:
    matrix = dict(payload.get("matrix", {}))
    required = SAFE_CONTROLLER_VARIANTS | COMPARISON_CONTROLLERS
    scenarios = list(payload.get("scenarios", []))
    return bool(scenarios) and all(required <= set(dict(matrix.get(scenario, {})).keys()) for scenario in scenarios)


def _matrix_has_pareto(payload: Dict[str, Any]) -> bool:
    matrix = dict(payload.get("matrix", {}))
    scenarios = list(payload.get("scenarios", []))
    return bool(scenarios) and all(bool(dict(matrix.get(scenario, {})).get("pareto_front")) for scenario in scenarios)


def _status(pass_condition: bool, partial_condition: bool = False) -> str:
    if pass_condition:
        return "pass"
    if partial_condition:
        return "partial"
    return "open"


def _criterion(
    criteria: list[dict[str, Any]],
    key: str,
    status: str,
    evidence: list[str],
    missing: list[str] | None = None,
) -> None:
    criteria.append(
        {
            "key": key,
            "status": status,
            "evidence": evidence,
            "missing": list(missing or []),
        }
    )


def analyze_theory(path: Path) -> Dict[str, Any]:
    payload, release_dir, mc100_payload = _load_release(path)
    novelty = analyze_novelty(path)
    criteria: list[dict[str, Any]] = []
    checks: Dict[str, Any] = {}
    warnings: List[str] = []

    motor_model = _source_contains(
        ROOT / "models" / "induction_motor_alpha_beta.py",
        ["AlphaBetaInductionMotorModel", "torque_nm", "randomized_motor_params", "_effective_lm"],
    )
    checks["motor_model_alpha_beta"] = motor_model
    _criterion(
        criteria,
        "alpha_beta_motor_model",
        _status(motor_model),
        ["models/induction_motor_alpha_beta.py"],
        [] if motor_model else ["alpha-beta flux/current/torque/randomization implementation"],
    )

    inverter_model = _source_contains(
        ROOT / "models" / "two_level_inverter.py",
        ["vector_bits", "alpha_beta_voltage", "common_mode_voltage", "estimate_inverter_losses"],
    )
    checks["two_level_inverter_model"] = inverter_model
    _criterion(
        criteria,
        "two_level_inverter_model",
        _status(inverter_model),
        ["models/two_level_inverter.py"],
        [] if inverter_model else ["legal-vector voltage/loss/common-mode implementation"],
    )

    fault = dict(payload.get("fault_injection", {}))
    no_deadtime = dict(dict(fault.get("cases", {})).get("no_deadtime_transition_emulation", {}))
    safety_ok = bool(
        fault.get("all_gateway_cases_no_shoot_through", False)
        and fault.get("raw_shoot_through_detector_triggered", False)
        and no_deadtime.get("direct_leg_transition_without_deadtime", False)
        and no_deadtime.get("safe_deadtime_path_valid", False)
    )
    checks["safety_gateway_timing_invariants"] = safety_ok
    _criterion(
        criteria,
        "safety_gateway_invariants",
        _status(safety_ok),
        [
            "safety/ai_pwm_gateway.py",
            "safe_neural_horizon_pwm_results.json:fault_injection",
        ],
        [] if safety_ok else ["no-shoot-through and no-direct-HIGH-to-LOW evidence"],
    )

    matrix = dict(payload.get("matrix", {}))
    scenarios = list(payload.get("scenarios", []))
    safe_variants = bool(scenarios) and all(
        SAFE_CONTROLLER_VARIANTS <= set(dict(matrix.get(scenario, {})).keys()) for scenario in scenarios
    )
    ablation = dict(payload.get("ablation", {}))
    ablation_ok = ABLATION_KEYS <= set(ablation.keys())
    checks["horizon_ai_pwm_variants"] = bool(safe_variants and ablation_ok)
    _criterion(
        criteria,
        "horizon_ai_pwm_variants",
        _status(bool(safe_variants and ablation_ok), bool(safe_variants or ablation_ok)),
        ["safe_neural_horizon_pwm_results.json:matrix", "safe_neural_horizon_pwm_results.json:ablation"],
        [] if safe_variants and ablation_ok else ["H2/H3/H4 variants and ablation variants"],
    )

    twin_scaffold = _source_contains(
        ROOT / "control" / "safe_neural_horizon_pwm.py",
        ["class NeuralTwin", "EventTriggeredFeedbackPolicy", "confidence", "residual_norm"],
    )
    checks["neural_twin_event_feedback_scaffold"] = twin_scaffold
    _criterion(
        criteria,
        "neural_twin_event_feedback_scaffold",
        _status(twin_scaffold),
        ["control/safe_neural_horizon_pwm.py"],
        [] if twin_scaffold else ["twin/confidence/event feedback implementation"],
    )

    comparison_matrix = _matrix_has_required_controllers(payload)
    checks["comparison_matrix"] = comparison_matrix
    checks["proxy_comparison_matrix"] = comparison_matrix
    _criterion(
        criteria,
        "comparison_matrix",
        _status(comparison_matrix),
        ["safe_neural_horizon_pwm_results.json:matrix"],
        [] if comparison_matrix else ["safe variants plus FOC-SVM/FCS-MPC/DTC/DTC-SVM baselines and remaining proxy controllers"],
    )

    missing_scenarios = [name for name in DEFAULT_SCENARIOS if name not in scenarios]
    robust_matrix = not missing_scenarios
    checks["robust_scenario_matrix"] = robust_matrix
    _criterion(
        criteria,
        "robust_scenario_matrix",
        _status(robust_matrix, bool(scenarios)),
        ["safe_neural_horizon_pwm_results.json:scenarios"],
        missing_scenarios,
    )

    mc100_ok = bool(mc100_payload and int(mc100_payload.get("mc_trials", 0)) >= 100)
    mc_small = int(payload.get("mc_trials", 0)) >= 3
    checks["first_mc100_smoke"] = mc100_ok
    _criterion(
        criteria,
        "first_mc100_smoke",
        _status(mc100_ok, mc_small),
        ["safe_neural_horizon_pwm_mc100_smoke.json" if mc100_payload else "safe_neural_horizon_pwm_results.json"],
        [] if mc100_ok else ["tracked MC>=100 host smoke evidence"],
    )

    pareto_ok = _matrix_has_pareto(payload) and bool(ablation.get("pareto_front"))
    checks["ablation_and_pareto_smoke"] = pareto_ok
    _criterion(
        criteria,
        "ablation_and_pareto_smoke",
        _status(pareto_ok, ablation_ok),
        ["safe_neural_horizon_pwm_results.json:ablation", "safe_neural_horizon_pwm_results.json:matrix[*].pareto_front"],
        [] if pareto_ok else ["Pareto fronts for every scenario and ablation"],
    )

    if release_dir is not None:
        report_files = [
            release_dir / "safe_neural_horizon_pwm_report.md",
            release_dir / "safe_neural_horizon_pwm_article_draft.md",
            release_dir / "WHAT_IS_NOT_DONE.md",
            release_dir / "figures" / "safe_neural_horizon_pwm_summary.csv",
        ]
        report_ok = all(path.exists() for path in report_files)
    else:
        report_files = []
        report_ok = False
    checks["report_and_release_artifacts"] = report_ok
    _criterion(
        criteria,
        "report_and_release_artifacts",
        _status(report_ok),
        [str(path.relative_to(release_dir)) for path in report_files] if release_dir is not None else [],
        [] if report_ok else ["tracked report/article/open-items/figures release package"],
    )

    honesty_ok = bool(
        novelty.get("host_novelty_claim_supported", False)
        and payload.get("hardware_claim") is False
        and "MCU/HIL/bench readiness" in list(novelty.get("not_allowed_claims", []))
    )
    checks["honest_claim_boundaries"] = honesty_ok
    _criterion(
        criteria,
        "honest_claim_boundaries",
        _status(honesty_ok),
        ["safe_neural_horizon_pwm_novelty_audit.json", "safe_neural_horizon_pwm_results.json:hardware_claim"],
        [] if honesty_ok else ["explicit not-allowed claims and hardware_claim=false"],
    )

    strong_baselines_ready = False
    foc_svm_key_baseline_ready = _source_contains(
        ROOT / "control" / "foc_svm_key_baseline.py",
        ["FocSvmKeyBaselineController", "_select_svm_vector", "alpha_beta_to_dq", "dq_to_alpha_beta"],
    ) and comparison_matrix
    fcs_mpc_one_step_baseline_ready = _source_contains(
        ROOT / "control" / "fcs_mpc_baseline.py",
        ["FcsMpcOneStepBaselineController", "_select_vector", "_score_vector", "candidate_torque"],
    ) and comparison_matrix
    dtc_hysteresis_baseline_ready = _source_contains(
        ROOT / "control" / "dtc_baseline.py",
        ["DtcHysteresisBaselineController", "_hysteresis", "torque_hysteresis_cmd", "flux_hysteresis_cmd"],
    ) and comparison_matrix
    dtc_svm_baseline_ready = _source_contains(
        ROOT / "control" / "dtc_svm_baseline.py",
        ["DtcSvmBaselineController", "_voltage_reference", "_select_svm_vector", "torque_error", "flux_error"],
    ) and comparison_matrix
    deadbeat_current_baseline_ready = _source_contains(
        ROOT / "control" / "deadbeat_current_baseline.py",
        ["DeadbeatCurrentBaselineController", "_deadbeat_voltage_ref", "_select_vector", "candidate_current_error"],
    ) and comparison_matrix
    trained_twin_ready = False
    publication_mc_ready = bool(mc100_payload and int(mc100_payload.get("mc_trials", 0)) >= 500)
    publication_plots_ready = False
    checks["foc_svm_key_baseline_ready"] = foc_svm_key_baseline_ready
    checks["fcs_mpc_one_step_baseline_ready"] = fcs_mpc_one_step_baseline_ready
    checks["dtc_hysteresis_baseline_ready"] = dtc_hysteresis_baseline_ready
    checks["dtc_svm_baseline_ready"] = dtc_svm_baseline_ready
    checks["deadbeat_current_baseline_ready"] = deadbeat_current_baseline_ready
    checks["strong_baselines_ready"] = strong_baselines_ready
    checks["trained_domain_randomized_twin_ready"] = trained_twin_ready
    checks["publication_mc500_ready"] = publication_mc_ready
    checks["publication_plots_fft_thd_ready"] = publication_plots_ready
    warnings.extend(
        [
            "FOC-SVM, one-step FCS-MPC, DTC hysteresis, DTC-SVM, and deadbeat current control have separate host baselines, but sensorless remains a proxy implementation",
            "neural twin is still a scaffold, not a trained domain-randomized model",
            "publication-scale MC and FFT/THD trace package are still open",
        ]
    )

    host_required = [
        "motor_model_alpha_beta",
        "two_level_inverter_model",
        "safety_gateway_timing_invariants",
        "horizon_ai_pwm_variants",
        "neural_twin_event_feedback_scaffold",
        "comparison_matrix",
        "robust_scenario_matrix",
        "first_mc100_smoke",
        "ablation_and_pareto_smoke",
        "report_and_release_artifacts",
        "honest_claim_boundaries",
    ]
    host_theory_scaffold_ready = all(bool(checks.get(key, False)) for key in host_required)
    publication_theory_complete = all(
        [
            host_theory_scaffold_ready,
            strong_baselines_ready,
            trained_twin_ready,
            publication_mc_ready,
            publication_plots_ready,
        ]
    )
    pass_count = sum(1 for item in criteria if item["status"] == "pass")
    partial_count = sum(1 for item in criteria if item["status"] == "partial")
    completion_pct = round(100.0 * pass_count / max(len(criteria), 1), 2)

    return {
        "status": "safe_neural_horizon_pwm_theory_completion_audit",
        "host_theory_scaffold_ready": host_theory_scaffold_ready,
        "publication_theory_complete": publication_theory_complete,
        "completion_pct_host_criteria": completion_pct,
        "criteria_total": len(criteria),
        "criteria_pass": pass_count,
        "criteria_partial": partial_count,
        "criteria_open": len(criteria) - pass_count - partial_count,
        "checks": checks,
        "criteria": criteria,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Safe Neural Horizon PWM theory completion evidence.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true", help="Fail if host_theory_scaffold_ready is false.")
    parser.add_argument("--publication-strict", action="store_true", help="Fail unless publication_theory_complete is true.")
    args = parser.parse_args()

    result = analyze_theory(Path(args.input).expanduser().resolve())
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out}")
    print(f"host_theory_scaffold_ready: {result['host_theory_scaffold_ready']}")
    print(f"publication_theory_complete: {result['publication_theory_complete']}")
    if args.strict and not bool(result["host_theory_scaffold_ready"]):
        raise SystemExit(1)
    if args.publication_strict and not bool(result["publication_theory_complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
