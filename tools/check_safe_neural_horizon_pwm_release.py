from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_safe_neural_horizon_pwm_study import DEFAULT_SCENARIOS


ALLOWED_FAULT_LATCH_SCENARIOS = {"fault_injection_runtime"}
REQUIRED_CONTROLLERS = {
    "protected_ai_pwm_h1_proxy",
    "fcs_mpc_one_step_proxy",
    "foc_svm_key_proxy",
    "dtc_hysteresis_proxy",
    "dtc_svm_proxy",
    "deadbeat_current_proxy",
    "sensorless_adaptive_foc_proxy",
    "safe_neural_horizon_pwm_h2",
    "safe_neural_horizon_pwm_h3_thermal",
    "safe_neural_horizon_pwm_h4_sparse",
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


def _check_manifest_hashes(release_dir: Path | None) -> tuple[bool, list[str]]:
    if release_dir is None:
        return True, []
    manifest_path = release_dir / "HOST_RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        return False, ["missing HOST_RELEASE_MANIFEST.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest.get("files", []):
        rel = Path(str(item.get("path", "")))
        path = release_dir / rel
        if not path.exists():
            failures.append(f"manifest file missing: {rel}")
            continue
        expected = str(item.get("sha256", ""))
        actual = _sha256(path)
        if actual != expected:
            failures.append(f"sha256 mismatch: {rel}")
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
    if not checks["fault_gateway_no_shoot_through"]:
        failures.append("gateway fault-injection cases include shoot-through")
    if not checks["raw_shoot_through_detector_triggered"]:
        failures.append("raw shoot-through detector did not trigger")

    manifest_ok, manifest_failures = _check_manifest_hashes(release_dir)
    checks["manifest_hashes_ok"] = manifest_ok
    if manifest_failures:
        failures.extend(manifest_failures)

    proxy_controllers = sorted(name for name in REQUIRED_CONTROLLERS if name.endswith("_proxy"))
    checks["strong_baselines_ready"] = False
    warnings.append(
        "strong baselines are not complete; current FOC-SVM/FCS-MPC/DTC/deadbeat/sensorless entries are proxy controllers"
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
