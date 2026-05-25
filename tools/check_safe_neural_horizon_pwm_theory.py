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
from tools.check_safe_neural_horizon_pwm_release import validate_mc_smoke_evidence
from tools.run_safe_neural_horizon_pwm_study import DEFAULT_SCENARIOS

TRACE_REQUIRED_CONTROLLERS = {
    "protected_ai_pwm_h1_baseline",
    "fcs_mpc_one_step_baseline",
    "foc_svm_key_baseline",
    "dtc_svm_baseline",
    "deadbeat_current_baseline",
    "sensorless_adaptive_foc_baseline",
    "safe_neural_horizon_pwm_h2",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release(path: Path) -> tuple[Dict[str, Any], Path | None, Dict[str, Any] | None, Dict[str, Any] | None]:
    if path.is_dir():
        result_path = path / "safe_neural_horizon_pwm_results.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        mc100_path = path / "safe_neural_horizon_pwm_mc100_smoke.json"
        mc500_path = path / "safe_neural_horizon_pwm_mc500_publication_smoke.json"
        return (
            _load_json(result_path),
            path,
            _load_json(mc100_path) if mc100_path.exists() else None,
            _load_json(mc500_path) if mc500_path.exists() else None,
        )
    return _load_json(path), None, None, None


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


def _trace_evidence_status(release_dir: Path | None) -> tuple[bool, bool, list[str]]:
    if release_dir is None:
        return False, False, ["release directory with trace_evidence/trace_summary.json"]
    trace_dir = release_dir / "trace_evidence"
    summary_path = trace_dir / "trace_summary.json"
    if not summary_path.exists():
        return False, False, ["trace_evidence/trace_summary.json"]
    payload = _load_json(summary_path)
    missing: list[str] = []
    if payload.get("hardware_claim") is not False:
        missing.append("hardware_claim=false")
    if not payload.get("trace_evidence_ready", False):
        missing.append("trace_evidence_ready=true")
    controllers = set(str(item) for item in payload.get("controllers", []))
    missing_controllers = sorted(TRACE_REQUIRED_CONTROLLERS - controllers)
    if missing_controllers:
        missing.append(f"trace controllers: {missing_controllers}")
    required_files = {
        "trace_summary.csv",
        "figures/fig_trace_speed.svg",
        "figures/fig_trace_fft_thd.svg",
    }
    for rel in sorted(required_files):
        if not (trace_dir / rel).exists():
            missing.append(f"trace file: {rel}")
    summary_rows = [dict(row) for row in payload.get("summary", [])]
    if len(summary_rows) < len(TRACE_REQUIRED_CONTROLLERS):
        missing.append("summary rows for required controllers")
    for row in summary_rows:
        for key in ("current_thd_like", "torque_thd_like", "current_rms", "torque_ripple_rms"):
            try:
                value = float(row.get(key, 0.0))
            except Exception:
                value = float("nan")
            if not (value >= 0.0 and value < float("inf")):
                missing.append(f"finite trace metric {row.get('controller', '?')}:{key}")
        if float(row.get("safety_violations", 1.0)) != 0.0:
            missing.append(f"trace safety violation: {row.get('controller', '?')}")
    evidence_ready = not missing
    publication_ready = evidence_ready and int(payload.get("steps", 0)) >= 512
    if evidence_ready and not publication_ready:
        missing.append("trace steps >= 512 for publication plot gate")
    return evidence_ready, publication_ready, missing


def _twin_evidence_status(release_dir: Path | None) -> tuple[bool, list[str]]:
    if release_dir is None:
        return False, ["release directory with twin_evidence/twin_training_summary.json"]
    twin_dir = release_dir / "twin_evidence"
    summary_path = twin_dir / "twin_training_summary.json"
    weights_path = twin_dir / "residual_twin_weights.json"
    if not summary_path.exists():
        return False, ["twin_evidence/twin_training_summary.json"]
    payload = _load_json(summary_path)
    missing: list[str] = []
    if payload.get("hardware_claim") is not False:
        missing.append("hardware_claim=false")
    if not payload.get("trained_domain_randomized_twin_ready", False):
        missing.append("trained_domain_randomized_twin_ready=true")
    if not payload.get("identified_domain_randomized_twin_ready", False):
        missing.append("identified_domain_randomized_twin_ready=true")
    if not weights_path.exists():
        missing.append("twin_evidence/residual_twin_weights.json")
    domain = dict(payload.get("domain_randomization", {}))
    for key in ("Rs", "Rr", "Lm", "J", "B"):
        if key not in domain:
            missing.append(f"domain_randomization.{key}")
    theta_multi = dict(payload.get("theta_conditioned_multi_step", {}))
    for horizon in ("1", "5", "10", "50"):
        row = dict(theta_multi.get(horizon, {}))
        if not row:
            missing.append(f"theta_conditioned_multi_step.{horizon}")
            continue
        try:
            improvement = float(row.get("improvement_pct", -1.0))
            rmse = float(row.get("theta_conditioned_twin_rmse", float("inf")))
        except Exception:
            improvement = -1.0
            rmse = float("inf")
        if improvement <= 0.0 or not (rmse >= 0.0 and rmse < float("inf")):
            missing.append(f"valid theta multi-step metric {horizon}")
    limits = [str(item).lower() for item in payload.get("interpretation_limits", [])]
    if not any("host simulation" in item for item in limits):
        missing.append("host simulation limitation")
    if not any("not mcu" in item or "hil" in item or "bench" in item for item in limits):
        missing.append("no MCU/HIL/bench limitation")
    return not missing, missing


def _baseline_strength_status(release_dir: Path | None) -> tuple[bool, bool, bool, bool, list[str]]:
    if release_dir is None:
        return False, False, False, False, ["release directory with safe_neural_horizon_pwm_baseline_strength_audit.json"]
    audit_path = release_dir / "safe_neural_horizon_pwm_baseline_strength_audit.json"
    if not audit_path.exists():
        return False, False, False, False, ["safe_neural_horizon_pwm_baseline_strength_audit.json"]
    payload = _load_json(audit_path)
    missing: list[str] = []
    if payload.get("hardware_claim") is not False:
        missing.append("hardware_claim=false")
    host_ready = bool(payload.get("host_baseline_scaffold_ready", False))
    publication_ready = bool(payload.get("publication_strong_baselines_ready", False))
    stress_ready = bool(payload.get("stress_evidence_ready", False))
    tuning_ready = bool(payload.get("tuning_evidence_ready", False))
    if not host_ready:
        missing.append("host_baseline_scaffold_ready=true")
    if not stress_ready:
        missing.append("stress_evidence_ready=true")
    if not tuning_ready:
        missing.append("tuning_evidence_ready=true")
    baselines = dict(payload.get("baselines", {}))
    for name, row_raw in baselines.items():
        row = dict(row_raw)
        if not bool(row.get("baseline_scaffold_ready", False)):
            missing.append(f"{name}: baseline_scaffold_ready")
    if not publication_ready:
        missing.append("publication_strong_baselines_ready=true after parameter-sweep tuning evidence")
    return host_ready, publication_ready, stress_ready, tuning_ready, missing


def _algorithm_identity_status(release_dir: Path | None) -> tuple[bool, list[str]]:
    if release_dir is None:
        return False, ["release directory with safe_neural_horizon_pwm_algorithm_identity_audit.json"]
    audit_path = release_dir / "safe_neural_horizon_pwm_algorithm_identity_audit.json"
    if not audit_path.exists():
        return False, ["safe_neural_horizon_pwm_algorithm_identity_audit.json"]
    payload = _load_json(audit_path)
    missing: list[str] = []
    if payload.get("hardware_claim") is not False:
        missing.append("hardware_claim=false")
    if not bool(payload.get("new_algorithm_identity_supported", False)):
        missing.append("new_algorithm_identity_supported=true")
    if len(list(payload.get("algorithm_identity_tuple", []))) < 5:
        missing.append("algorithm_identity_tuple length >= 5")
    features = dict(payload.get("essential_features", {}))
    for name, row_raw in features.items():
        row = dict(row_raw)
        if not bool(row.get("ready", False)):
            missing.append(f"{name}: ready")
    return not missing, missing


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
    payload, release_dir, mc100_payload, mc500_payload = _load_release(path)
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
    checks["named_baseline_comparison_matrix"] = comparison_matrix
    _criterion(
        criteria,
        "comparison_matrix",
        _status(comparison_matrix),
        ["safe_neural_horizon_pwm_results.json:matrix"],
        [] if comparison_matrix else ["safe variants plus FOC-SVM/FCS-MPC/DTC/DTC-SVM/deadbeat/sensorless baselines"],
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

    mc100_ok, mc100_missing = validate_mc_smoke_evidence(mc100_payload, min_trials=100, label="MC100")
    mc_small = int(payload.get("mc_trials", 0)) >= 3
    checks["first_mc100_smoke"] = mc100_ok
    _criterion(
        criteria,
        "first_mc100_smoke",
        _status(mc100_ok, mc_small),
        ["safe_neural_horizon_pwm_mc100_smoke.json" if mc100_payload else "safe_neural_horizon_pwm_results.json"],
        [] if mc100_ok else mc100_missing,
    )

    mc500_ok, mc500_missing = validate_mc_smoke_evidence(mc500_payload, min_trials=500, label="MC500")
    checks["publication_mc500_ready"] = mc500_ok
    _criterion(
        criteria,
        "publication_mc500_evidence",
        _status(mc500_ok, mc100_ok),
        ["safe_neural_horizon_pwm_mc500_publication_smoke.json" if mc500_payload else "safe_neural_horizon_pwm_mc100_smoke.json"],
        [] if mc500_ok else mc500_missing,
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

    trace_evidence_ready, publication_trace_ready, trace_missing = _trace_evidence_status(release_dir)
    checks["trace_fft_thd_evidence_ready"] = trace_evidence_ready
    checks["publication_plots_fft_thd_ready"] = publication_trace_ready
    _criterion(
        criteria,
        "trace_fft_thd_evidence",
        _status(trace_evidence_ready, release_dir is not None),
        ["trace_evidence/trace_summary.json", "trace_evidence/trace_summary.csv", "trace_evidence/figures/*.svg"],
        [] if trace_evidence_ready else trace_missing,
    )

    twin_evidence_ready, twin_missing = _twin_evidence_status(release_dir)
    checks["domain_randomized_twin_evidence_ready"] = twin_evidence_ready
    _criterion(
        criteria,
        "domain_randomized_twin_evidence",
        _status(twin_evidence_ready, release_dir is not None),
        ["twin_evidence/twin_training_summary.json", "twin_evidence/residual_twin_weights.json"],
        [] if twin_evidence_ready else twin_missing,
    )

    (
        baseline_scaffold_ready,
        publication_baselines_ready,
        baseline_stress_ready,
        baseline_tuning_ready,
        baseline_missing,
    ) = _baseline_strength_status(release_dir)
    checks["baseline_strength_audit_ready"] = baseline_scaffold_ready
    checks["baseline_stress_evidence_ready"] = baseline_stress_ready
    checks["baseline_tuning_evidence_ready"] = baseline_tuning_ready
    _criterion(
        criteria,
        "baseline_strength_audit",
        _status(baseline_scaffold_ready, release_dir is not None),
        ["safe_neural_horizon_pwm_baseline_strength_audit.json"],
        [] if baseline_scaffold_ready else baseline_missing,
    )

    algorithm_identity_ready, algorithm_identity_missing = _algorithm_identity_status(release_dir)
    checks["algorithm_identity_ready"] = algorithm_identity_ready
    _criterion(
        criteria,
        "algorithm_identity_audit",
        _status(algorithm_identity_ready, release_dir is not None),
        ["safe_neural_horizon_pwm_algorithm_identity_audit.json"],
        [] if algorithm_identity_ready else algorithm_identity_missing,
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

    strong_baselines_ready = publication_baselines_ready
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
    sensorless_adaptive_foc_baseline_ready = _source_contains(
        ROOT / "control" / "sensorless_adaptive_foc_baseline.py",
        ["SensorlessAdaptiveFocBaselineController", "_mras_speed_update", "_adapt_rs", "omega_hat"],
    ) and comparison_matrix
    protected_ai_pwm_h1_baseline_ready = _source_contains(
        ROOT / "control" / "protected_ai_pwm_h1_baseline.py",
        ["ProtectedAiPwmH1BaselineController", "protected_h1_config", "horizon=1"],
    ) and comparison_matrix
    trained_twin_ready = twin_evidence_ready
    publication_mc_ready = mc500_ok
    publication_plots_ready = publication_trace_ready
    checks["foc_svm_key_baseline_ready"] = foc_svm_key_baseline_ready
    checks["fcs_mpc_one_step_baseline_ready"] = fcs_mpc_one_step_baseline_ready
    checks["dtc_hysteresis_baseline_ready"] = dtc_hysteresis_baseline_ready
    checks["dtc_svm_baseline_ready"] = dtc_svm_baseline_ready
    checks["deadbeat_current_baseline_ready"] = deadbeat_current_baseline_ready
    checks["sensorless_adaptive_foc_baseline_ready"] = sensorless_adaptive_foc_baseline_ready
    checks["protected_ai_pwm_h1_baseline_ready"] = protected_ai_pwm_h1_baseline_ready
    checks["strong_baselines_ready"] = strong_baselines_ready
    checks["trained_domain_randomized_twin_ready"] = trained_twin_ready
    checks["publication_mc500_ready"] = publication_mc_ready
    checks["publication_plots_fft_thd_ready"] = publication_plots_ready
    warnings.extend(
        [
            "FOC-SVM, one-step FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC have bounded host tuning evidence" if strong_baselines_ready else "FOC-SVM, one-step FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC have separate host baselines, but are not final publication-tuned",
            "domain-randomized twin evidence is host-only and theta-conditioned; production online identification remains open",
            "publication-scale MC500 host evidence exists" if publication_mc_ready else "publication-scale MC is still open",
        ]
    )
    if not publication_trace_ready:
        warnings.append("publication trace/FFT/THD plot gate is still open")

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
        "baseline_strength_audit_ready",
        "report_and_release_artifacts",
        "honest_claim_boundaries",
        "algorithm_identity_ready",
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
