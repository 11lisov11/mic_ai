from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_safe_neural_horizon_pwm_study import DEFAULT_SCENARIOS


ALLOWED_FAULT_LATCH_SCENARIOS = {"fault_injection_runtime"}
REQUIRED_CONTROLLERS = {
    "protected_ai_pwm_h1_baseline",
    "fcs_mpc_one_step_baseline",
    "foc_svm_key_baseline",
    "dtc_hysteresis_baseline",
    "dtc_svm_baseline",
    "deadbeat_current_baseline",
    "sensorless_adaptive_foc_baseline",
    "safe_neural_horizon_pwm_h2",
    "safe_neural_horizon_pwm_h3_thermal",
    "safe_neural_horizon_pwm_h4_sparse",
}
REQUIRED_RELEASE_FILES = {
    "safe_neural_horizon_pwm_results.json",
    "safe_neural_horizon_pwm_report.md",
    "safe_neural_horizon_pwm_article_draft.md",
    "safe_neural_horizon_pwm_baseline_stress_evidence.json",
    "safe_neural_horizon_pwm_baseline_tuning_evidence.json",
    "safe_neural_horizon_pwm_baseline_strength_audit.json",
    "safe_neural_horizon_pwm_algorithm_identity_audit.json",
    "safe_neural_horizon_pwm_novelty_audit.json",
    "safe_neural_horizon_pwm_theory_completion_audit.json",
    "safe_neural_horizon_pwm_mc100_smoke.json",
    "safe_neural_horizon_pwm_mc500_publication_smoke.json",
    "WHAT_IS_NOT_DONE.md",
    "figures/safe_neural_horizon_pwm_summary.csv",
    "figures/fig_speed_error_vs_current.svg",
    "figures/fig_feedback_vs_switching.svg",
    "figures/fig_h2_scenario_speed_error.svg",
}
MC_SMOKE_REQUIRED_CONTROLLERS = {
    "protected_ai_pwm_h1_baseline",
    "fcs_mpc_one_step_baseline",
    "foc_svm_key_baseline",
    "dtc_hysteresis_baseline",
    "dtc_svm_baseline",
    "deadbeat_current_baseline",
    "sensorless_adaptive_foc_baseline",
    "safe_neural_horizon_pwm_h2",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric(row: Dict[str, Any], name: str, field: str = "worst") -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get(field, 0.0))
    return float(value or 0.0)


def _load_results(path: Path) -> tuple[Dict[str, Any], Path | None]:
    if path.is_dir():
        result_path = path / "safe_neural_horizon_pwm_results.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        return json.loads(result_path.read_text(encoding="utf-8")), path
    return json.loads(path.read_text(encoding="utf-8")), None


def _normalize_manifest_path(raw: str) -> tuple[str | None, str | None]:
    if not raw:
        return None, "empty manifest path"
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or Path(raw).is_absolute():
        return None, f"unsafe absolute manifest path: {raw}"
    rel = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} or ":" in part for part in rel.parts):
        return None, f"unsafe manifest path: {raw}"
    return rel.as_posix(), None


def _check_manifest_hashes(release_dir: Path | None) -> tuple[bool, bool, bool, list[str]]:
    if release_dir is None:
        return True, True, True, []
    manifest_path = release_dir / "HOST_RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        return False, False, False, ["missing HOST_RELEASE_MANIFEST.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    manifest_paths_safe = True
    manifest_entries: set[str] = set()
    for item in manifest.get("files", []):
        raw = str(item.get("path", ""))
        rel_key, rel_error = _normalize_manifest_path(raw)
        if rel_error:
            manifest_paths_safe = False
            failures.append(rel_error)
            continue
        assert rel_key is not None
        manifest_entries.add(rel_key)
        path = release_dir.joinpath(*PurePosixPath(rel_key).parts)
        if not path.exists():
            failures.append(f"manifest file missing: {rel_key}")
            continue
        expected = str(item.get("sha256", ""))
        actual = _sha256(path)
        if actual != expected:
            failures.append(f"sha256 mismatch: {rel_key}")
    missing_required = sorted(REQUIRED_RELEASE_FILES - manifest_entries)
    required_files_present = not missing_required
    if missing_required:
        failures.append(f"manifest missing required release files: {missing_required}")
    return not failures, required_files_present, manifest_paths_safe, failures


def validate_mc_smoke_evidence(payload: Dict[str, Any] | None, *, min_trials: int, label: str) -> tuple[bool, list[str]]:
    """Validate that tracked MC evidence is a real host run, not only a trial counter."""
    if payload is None:
        return False, [f"{label}: missing payload"]
    failures: list[str] = []
    if payload.get("hardware_claim") is not False:
        failures.append(f"{label}: hardware_claim must be false")
    status = str(payload.get("status", ""))
    if not status.startswith("host_") and status != "HOST_SIMULATION_ONLY":
        failures.append(f"{label}: status must be host-only")
    try:
        trials = int(payload.get("mc_trials", 0))
    except Exception:
        trials = 0
    if trials < min_trials:
        failures.append(f"{label}: mc_trials must be >= {min_trials}")
    try:
        steps = int(payload.get("steps_per_trial", 0))
    except Exception:
        steps = 0
    if steps <= 0:
        failures.append(f"{label}: steps_per_trial must be positive")
    controllers = dict(payload.get("controllers", {}))
    missing = sorted(MC_SMOKE_REQUIRED_CONTROLLERS - set(controllers.keys()))
    if missing:
        failures.append(f"{label}: missing controllers {missing}")
    for controller in sorted(MC_SMOKE_REQUIRED_CONTROLLERS & set(controllers.keys())):
        row = dict(controllers.get(controller, {}))
        safety = _metric(row, "safety_violations", "worst")
        if safety != 0.0:
            failures.append(f"{label}: safety violations for {controller}: {safety}")
        try:
            failure_count = int(row.get("failure_count", 0))
        except Exception:
            failure_count = 0
        if failure_count != 0:
            failures.append(f"{label}: failure_count for {controller}: {failure_count}")
    return not failures, failures


def analyze_release(path: Path) -> Dict[str, Any]:
    payload, release_dir = _load_results(path)
    checks: Dict[str, Any] = {}
    failures: List[str] = []
    warnings: List[str] = []

    checks["hardware_claim_false"] = bool(payload.get("hardware_claim", True) is False)
    if not checks["hardware_claim_false"]:
        failures.append("hardware_claim must be false for host release")

    status = str(payload.get("status", ""))
    checks["status_is_host"] = status.startswith("host_") or status == "HOST_SIMULATION_ONLY"
    if not checks["status_is_host"]:
        failures.append(f"unexpected status: {status}")

    scenarios = list(payload.get("scenarios", []))
    missing_scenarios = [name for name in DEFAULT_SCENARIOS if name not in scenarios]
    checks["required_scenarios_present"] = not missing_scenarios
    checks["scenario_count"] = len(scenarios)
    if missing_scenarios:
        failures.append(f"missing scenarios: {missing_scenarios}")

    matrix = dict(payload.get("matrix", {}))
    checks["matrix_present"] = bool(matrix)
    if not matrix:
        failures.append("matrix is missing")

    missing_controllers: Dict[str, list[str]] = {}
    h2_safety_failures: Dict[str, float] = {}
    h2_unexpected_fault_failures: Dict[str, int] = {}
    for scenario in scenarios:
        rows = dict(matrix.get(scenario, {}))
        missing = sorted(REQUIRED_CONTROLLERS - set(rows.keys()))
        if missing:
            missing_controllers[scenario] = missing
        h2 = dict(rows.get("safe_neural_horizon_pwm_h2", {}))
        safety_worst = _metric(h2, "safety_violations", "worst")
        if safety_worst != 0.0:
            h2_safety_failures[scenario] = safety_worst
        failure_count = int(h2.get("failure_count", 0))
        if failure_count and scenario not in ALLOWED_FAULT_LATCH_SCENARIOS:
            h2_unexpected_fault_failures[scenario] = failure_count

    checks["required_controllers_present"] = not missing_controllers
    checks["h2_no_safety_violations"] = not h2_safety_failures
    checks["h2_no_unexpected_fault_latches"] = not h2_unexpected_fault_failures
    if missing_controllers:
        failures.append(f"missing controllers in matrix: {missing_controllers}")
    if h2_safety_failures:
        failures.append(f"H2 safety violations: {h2_safety_failures}")
    if h2_unexpected_fault_failures:
        failures.append(f"H2 unexpected fault/failure counts: {h2_unexpected_fault_failures}")

    fault = dict(payload.get("fault_injection", {}))
    checks["fault_gateway_no_shoot_through"] = bool(fault.get("all_gateway_cases_no_shoot_through", False))
    checks["raw_shoot_through_detector_triggered"] = bool(fault.get("raw_shoot_through_detector_triggered", False))
    no_deadtime = dict(dict(fault.get("cases", {})).get("no_deadtime_transition_emulation", {}))
    checks["deadtime_transition_detector_triggered"] = bool(
        no_deadtime.get("direct_leg_transition_without_deadtime", False)
        and no_deadtime.get("safe_deadtime_path_valid", False)
        and no_deadtime.get("blocked_by_gateway_deadtime_path", False)
    )
    if not checks["fault_gateway_no_shoot_through"]:
        failures.append("gateway fault-injection cases include shoot-through")
    if not checks["raw_shoot_through_detector_triggered"]:
        failures.append("raw shoot-through detector did not trigger")
    if not checks["deadtime_transition_detector_triggered"]:
        failures.append("dead-time transition detector did not trigger")

    manifest_ok, required_release_files_present, manifest_paths_safe, manifest_failures = _check_manifest_hashes(release_dir)
    checks["manifest_hashes_ok"] = manifest_ok
    checks["required_release_files_present"] = required_release_files_present
    checks["manifest_paths_safe"] = manifest_paths_safe
    if manifest_failures:
        failures.extend(manifest_failures)

    if release_dir is not None:
        acceptance_path = release_dir / "HOST_ACCEPTANCE_SUMMARY.json"
        checks["acceptance_summary_present"] = acceptance_path.exists()
        if not checks["acceptance_summary_present"]:
            failures.append("missing HOST_ACCEPTANCE_SUMMARY.json")

        novelty_path = release_dir / "safe_neural_horizon_pwm_novelty_audit.json"
        identity_path = release_dir / "safe_neural_horizon_pwm_algorithm_identity_audit.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            checks["algorithm_identity_supported"] = bool(identity.get("new_algorithm_identity_supported", False))
            if not checks["algorithm_identity_supported"]:
                failures.append("algorithm identity audit does not support the new host-level control-law identity")
        else:
            checks["algorithm_identity_supported"] = False
            failures.append("missing safe_neural_horizon_pwm_algorithm_identity_audit.json")
        if novelty_path.exists():
            novelty = json.loads(novelty_path.read_text(encoding="utf-8"))
            checks["novelty_audit_supported"] = bool(novelty.get("host_novelty_claim_supported", False))
            if not checks["novelty_audit_supported"]:
                failures.append("novelty audit does not support the host-level novelty claim")
        else:
            checks["novelty_audit_supported"] = False
            failures.append("missing safe_neural_horizon_pwm_novelty_audit.json")
        baseline_path = release_dir / "safe_neural_horizon_pwm_baseline_strength_audit.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            checks["baseline_strength_audit_ready"] = bool(baseline.get("host_baseline_scaffold_ready", False))
            checks["baseline_stress_evidence_ready"] = bool(baseline.get("stress_evidence_ready", False))
            checks["baseline_tuning_evidence_ready"] = bool(baseline.get("tuning_evidence_ready", False))
            checks["publication_strong_baselines_ready"] = bool(
                baseline.get("publication_strong_baselines_ready", False)
            )
            if not checks["baseline_strength_audit_ready"]:
                failures.append("baseline strength audit does not support the host baseline scaffold")
            if not checks["baseline_stress_evidence_ready"]:
                failures.append("baseline strength audit does not support baseline stress evidence")
            if not checks["baseline_tuning_evidence_ready"]:
                failures.append("baseline strength audit does not support baseline tuning evidence")
        else:
            checks["baseline_strength_audit_ready"] = False
            checks["baseline_stress_evidence_ready"] = False
            checks["baseline_tuning_evidence_ready"] = False
            checks["publication_strong_baselines_ready"] = False
            failures.append("missing safe_neural_horizon_pwm_baseline_strength_audit.json")
        theory_path = release_dir / "safe_neural_horizon_pwm_theory_completion_audit.json"
        if theory_path.exists():
            theory = json.loads(theory_path.read_text(encoding="utf-8"))
            checks["theory_scaffold_ready"] = bool(theory.get("host_theory_scaffold_ready", False))
            checks["publication_theory_complete"] = bool(theory.get("publication_theory_complete", False))
            if not checks["theory_scaffold_ready"]:
                failures.append("theory completion audit does not support the host theory scaffold")
        else:
            checks["theory_scaffold_ready"] = False
            checks["publication_theory_complete"] = False
            failures.append("missing safe_neural_horizon_pwm_theory_completion_audit.json")

        mc100_path = release_dir / "safe_neural_horizon_pwm_mc100_smoke.json"
        mc100_payload = json.loads(mc100_path.read_text(encoding="utf-8")) if mc100_path.exists() else None
        mc100_ok, mc100_failures = validate_mc_smoke_evidence(mc100_payload, min_trials=100, label="MC100")
        checks["mc100_smoke_content_ready"] = mc100_ok
        if mc100_failures:
            failures.extend(mc100_failures)

        mc500_path = release_dir / "safe_neural_horizon_pwm_mc500_publication_smoke.json"
        mc500_payload = json.loads(mc500_path.read_text(encoding="utf-8")) if mc500_path.exists() else None
        mc500_ok, mc500_failures = validate_mc_smoke_evidence(mc500_payload, min_trials=500, label="MC500")
        checks["mc500_publication_content_ready"] = mc500_ok
        if mc500_failures:
            failures.extend(mc500_failures)

    proxy_controllers = sorted(name for name in REQUIRED_CONTROLLERS if name.endswith("_proxy"))
    checks["foc_svm_key_baseline_ready"] = bool(scenarios) and all(
        "foc_svm_key_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["fcs_mpc_one_step_baseline_ready"] = bool(scenarios) and all(
        "fcs_mpc_one_step_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["dtc_hysteresis_baseline_ready"] = bool(scenarios) and all(
        "dtc_hysteresis_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["dtc_svm_baseline_ready"] = bool(scenarios) and all(
        "dtc_svm_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["deadbeat_current_baseline_ready"] = bool(scenarios) and all(
        "deadbeat_current_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["sensorless_adaptive_foc_baseline_ready"] = bool(scenarios) and all(
        "sensorless_adaptive_foc_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["protected_ai_pwm_h1_baseline_ready"] = bool(scenarios) and all(
        "protected_ai_pwm_h1_baseline" in dict(matrix.get(scenario, {})) for scenario in scenarios
    )
    checks["strong_baselines_ready"] = bool(checks.get("publication_strong_baselines_ready", False))
    if checks["strong_baselines_ready"]:
        warnings.append(
            "strong baselines have bounded host parameter-sweep evidence; this is still not MCU/HIL/bench or universal-superiority evidence"
        )
    else:
        warnings.append(
            "strong baselines are not complete; FOC-SVM, one-step FCS-MPC, DTC hysteresis, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC are host baselines, but not final publication-tuned"
        )

    host_release_ready = not failures
    return {
        "status": "safe_neural_horizon_pwm_host_release_check",
        "host_release_ready": host_release_ready,
        "hardware_ready": False,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "proxy_controllers": proxy_controllers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Safe Neural Horizon PWM host-release evidence.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if host_release_ready is false.")
    args = parser.parse_args()

    result = analyze_release(Path(args.input).expanduser().resolve())
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out}")
    print(f"host_release_ready: {result['host_release_ready']}")
    if args.strict and not bool(result["host_release_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
