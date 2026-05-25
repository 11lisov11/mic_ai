from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_safe_neural_horizon_pwm_novelty import COMPARISON_CONTROLLERS


ALLOWED_FAULT_SCENARIOS = {"fault_injection_runtime"}
BASELINE_SOURCE_MARKERS = {
    "protected_ai_pwm_h1_baseline": ("control/protected_ai_pwm_h1_baseline.py", "ProtectedAiPwmH1BaselineController"),
    "fcs_mpc_one_step_baseline": ("control/fcs_mpc_baseline.py", "FcsMpcOneStepBaselineController"),
    "foc_svm_key_baseline": ("control/foc_svm_key_baseline.py", "FocSvmKeyBaselineController"),
    "dtc_hysteresis_baseline": ("control/dtc_baseline.py", "DtcHysteresisBaselineController"),
    "dtc_svm_baseline": ("control/dtc_svm_baseline.py", "DtcSvmBaselineController"),
    "deadbeat_current_baseline": ("control/deadbeat_current_baseline.py", "DeadbeatCurrentBaselineController"),
    "sensorless_adaptive_foc_baseline": (
        "control/sensorless_adaptive_foc_baseline.py",
        "SensorlessAdaptiveFocBaselineController",
    ),
}


def _load_payload(path: Path) -> tuple[Dict[str, Any], Path | None]:
    if path.is_dir():
        result_path = path / "safe_neural_horizon_pwm_results.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        return json.loads(result_path.read_text(encoding="utf-8")), path
    return json.loads(path.read_text(encoding="utf-8")), None


def _metric(row: Dict[str, Any], name: str, field: str = "mean") -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get(field, 0.0))
    return float(value or 0.0)


def _source_has_marker(controller: str) -> bool:
    rel, marker = BASELINE_SOURCE_MARKERS[controller]
    path = ROOT / rel
    return path.exists() and marker in path.read_text(encoding="utf-8")


def _trace_controllers(release_dir: Path | None) -> set[str]:
    if release_dir is None:
        return set()
    summary_path = release_dir / "trace_evidence" / "trace_summary.json"
    if not summary_path.exists():
        return set()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {str(item) for item in payload.get("controllers", [])}


def analyze_baselines(path: Path) -> Dict[str, Any]:
    payload, release_dir = _load_payload(path)
    matrix = dict(payload.get("matrix", {}))
    scenarios = [str(name) for name in payload.get("scenarios", [])]
    trace_covered = _trace_controllers(release_dir)
    failures: list[str] = []
    warnings: list[str] = []
    rows: Dict[str, Any] = {}

    for controller in sorted(COMPARISON_CONTROLLERS):
        missing_scenarios: list[str] = []
        safety_worst = 0.0
        unexpected_failure_count = 0
        pareto_scenarios: list[str] = []
        max_speed_mean = 0.0
        max_current_mean = 0.0
        max_switch_mean = 0.0
        finite_metrics = True

        for scenario in scenarios:
            scenario_row = dict(matrix.get(scenario, {}))
            if controller not in scenario_row:
                missing_scenarios.append(scenario)
                continue
            row = dict(scenario_row[controller])
            if controller in list(scenario_row.get("pareto_front", [])):
                pareto_scenarios.append(scenario)
            try:
                safety_worst = max(safety_worst, _metric(row, "safety_violations", "worst"))
                max_speed_mean = max(max_speed_mean, _metric(row, "mean_abs_speed_error", "mean"))
                max_current_mean = max(max_current_mean, _metric(row, "mean_current_abs", "mean"))
                max_switch_mean = max(max_switch_mean, _metric(row, "switch_events", "mean"))
            except Exception:
                finite_metrics = False
            if scenario not in ALLOWED_FAULT_SCENARIOS:
                try:
                    unexpected_failure_count += int(row.get("failure_count", 0))
                except Exception:
                    unexpected_failure_count += 1

        source_marker_present = _source_has_marker(controller)
        trace_present = controller in trace_covered
        matrix_coverage_ready = bool(scenarios) and not missing_scenarios
        safety_ready = safety_worst == 0.0 and unexpected_failure_count == 0
        pareto_ready = bool(pareto_scenarios)
        baseline_scaffold_ready = all(
            [source_marker_present, matrix_coverage_ready, safety_ready, pareto_ready, finite_metrics]
        )
        # This release still has no parameter-sweep/tuned-controller evidence file.
        publication_tuned_ready = False
        if not baseline_scaffold_ready:
            failures.append(f"{controller}: baseline scaffold is incomplete")
        if not publication_tuned_ready:
            warnings.append(f"{controller}: no publication-grade tuning/stress evidence yet")

        rows[controller] = {
            "source_marker_present": source_marker_present,
            "matrix_coverage_ready": matrix_coverage_ready,
            "missing_scenarios": missing_scenarios,
            "safety_ready": safety_ready,
            "safety_violations_worst": safety_worst,
            "unexpected_failure_count": unexpected_failure_count,
            "pareto_participation_count": len(pareto_scenarios),
            "pareto_scenarios": pareto_scenarios,
            "trace_present": trace_present,
            "finite_metrics": finite_metrics,
            "max_mean_abs_speed_error": max_speed_mean,
            "max_mean_current_abs": max_current_mean,
            "max_mean_switch_events": max_switch_mean,
            "baseline_scaffold_ready": baseline_scaffold_ready,
            "publication_tuned_ready": publication_tuned_ready,
        }

    required = set(COMPARISON_CONTROLLERS)
    present = set(rows.keys())
    missing = sorted(required - present)
    if missing:
        failures.append(f"missing baseline rows: {missing}")

    host_baseline_scaffold_ready = not failures and all(bool(row["baseline_scaffold_ready"]) for row in rows.values())
    publication_strong_baselines_ready = host_baseline_scaffold_ready and all(
        bool(row["publication_tuned_ready"]) for row in rows.values()
    )
    return {
        "status": "safe_neural_horizon_pwm_baseline_strength_audit",
        "hardware_claim": False,
        "host_baseline_scaffold_ready": host_baseline_scaffold_ready,
        "publication_strong_baselines_ready": publication_strong_baselines_ready,
        "baseline_count": len(rows),
        "scenario_count": len(scenarios),
        "baselines": rows,
        "failures": failures,
        "warnings": warnings,
        "interpretation": (
            "The comparison baselines are separate safe host implementations with scenario coverage and Pareto "
            "participation, but they are not yet publication-tuned strong baselines."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SNH-PWM comparison baseline strength.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true", help="Fail unless host_baseline_scaffold_ready is true.")
    parser.add_argument(
        "--publication-strict",
        action="store_true",
        help="Fail unless publication_strong_baselines_ready is true.",
    )
    args = parser.parse_args()

    result = analyze_baselines(Path(args.input).expanduser().resolve())
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out}")
    print(f"host_baseline_scaffold_ready: {result['host_baseline_scaffold_ready']}")
    print(f"publication_strong_baselines_ready: {result['publication_strong_baselines_ready']}")
    if args.strict and not bool(result["host_baseline_scaffold_ready"]):
        raise SystemExit(1)
    if args.publication_strict and not bool(result["publication_strong_baselines_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
