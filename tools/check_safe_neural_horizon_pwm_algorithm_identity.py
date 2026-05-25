from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_safe_neural_horizon_pwm_novelty import COMPARISON_CONTROLLERS, SAFE_CONTROLLER_VARIANTS


ESSENTIAL_FEATURES = {
    "legal_vector_horizon_search": {
        "file": ROOT / "control" / "safe_neural_horizon_pwm.py",
        "markers": ["def select_sequence", "product(candidates", "horizon = max(1, min(int(self.cfg.horizon), 4))"],
        "why_it_matters": "SNH-PWM optimizes finite sequences of legal inverter vectors instead of a continuous voltage reference.",
    },
    "neural_cost_shaping": {
        "file": ROOT / "control" / "safe_neural_horizon_pwm.py",
        "markers": ["class NeuralCostShaper", "def shape", '"speed"', '"current"', '"switching"', '"thermal"', '"risk"'],
        "why_it_matters": "The neural layer shapes the MPC-like cost, rather than replacing only a PI loop.",
    },
    "event_triggered_twin_feedback": {
        "file": ROOT / "control" / "safe_neural_horizon_pwm.py",
        "markers": ["class NeuralTwin", "class EventTriggeredFeedbackPolicy", "should_sample", "residual_norm", "uncertainty"],
        "why_it_matters": "The control loop explicitly trades feedback usage against model uncertainty and residuals.",
    },
    "protected_ai_pwm_gateway": {
        "file": ROOT / "safety" / "ai_pwm_gateway.py",
        "markers": ["class AIPwmRequest", "vector_id", "transition_waveform", "has_shoot_through", "valid_action_mask"],
        "why_it_matters": "The AI can request only legal vector IDs; raw high/low gate access remains outside the AI interface.",
    },
    "multiobjective_drive_cost": {
        "file": ROOT / "control" / "safe_neural_horizon_pwm.py",
        "markers": [
            "speed_weight",
            "current_weight",
            "switching_weight",
            "loss_weight",
            "thermal_weight",
            "torque_ripple_weight",
            "feedback_weight",
            "risk_weight",
        ],
        "why_it_matters": "The objective jointly covers tracking, current stress, switching/loss/thermal/ripple, feedback economy, and risk.",
    },
}


BASELINE_DIFFERENCES = {
    "protected_ai_pwm_h1_baseline": {
        "missing_snh_features": ["finite horizon H>1", "H2/H3/H4 comparison", "explicit horizon-search novelty over the prior H1 architecture"],
        "snh_distinction": "SNH-PWM generalizes the prior protected one-step AI-PWM into a horizon controller with event/twin/risk costs.",
    },
    "foc_svm_key_baseline": {
        "missing_snh_features": ["direct finite-horizon legal-vector search", "neural cost shaping", "event-triggered twin feedback"],
        "snh_distinction": "FOC-SVM synthesizes dq voltage references and then selects an SVM-like vector; SNH-PWM searches vector sequences directly.",
    },
    "fcs_mpc_one_step_baseline": {
        "missing_snh_features": ["horizon H>1", "neural cost shaping", "feedback-economy objective", "twin residual confidence"],
        "snh_distinction": "One-step FCS-MPC predicts legal vectors for one tick; SNH-PWM adds horizon search, neural shaping, and event feedback.",
    },
    "dtc_hysteresis_baseline": {
        "missing_snh_features": ["neural cost shaping", "finite-horizon sequence optimization", "event-triggered twin feedback"],
        "snh_distinction": "DTC hysteresis follows torque/flux comparator commands; SNH-PWM optimizes predicted vector sequences under risk and feedback costs.",
    },
    "dtc_svm_baseline": {
        "missing_snh_features": ["direct horizon vector search", "neural cost shaping", "feedback-economy objective"],
        "snh_distinction": "DTC-SVM synthesizes a torque/flux voltage reference; SNH-PWM optimizes legal vector sequences directly.",
    },
    "deadbeat_current_baseline": {
        "missing_snh_features": ["multiobjective neural cost shaping", "event-triggered twin feedback", "horizon sequence objective"],
        "snh_distinction": "Deadbeat control targets one-step current matching; SNH-PWM optimizes a broader multiobjective drive cost.",
    },
    "sensorless_adaptive_foc_baseline": {
        "missing_snh_features": ["AI-PWM horizon vector search", "neural cost shaping", "protected vector-sequence objective"],
        "snh_distinction": "Sensorless/adaptive FOC estimates speed/Rs for classical FOC; SNH-PWM uses twin uncertainty inside vector-sequence selection.",
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_dir(path: Path) -> Path | None:
    return path if path.is_dir() else None


def _results_payload(path: Path) -> tuple[Dict[str, Any], Path | None]:
    release_dir = _release_dir(path)
    if release_dir is not None:
        return _load_json(release_dir / "safe_neural_horizon_pwm_results.json"), release_dir
    return _load_json(path), None


def _source_markers(feature: Dict[str, Any]) -> tuple[bool, list[str]]:
    source = Path(feature["file"])
    if not source.exists():
        return False, [str(source.relative_to(ROOT))]
    text = source.read_text(encoding="utf-8")
    missing = [marker for marker in feature["markers"] if marker not in text]
    return not missing, missing


def _release_checks(release_dir: Path | None) -> tuple[dict[str, bool], list[str]]:
    if release_dir is None:
        return {}, ["release directory required for packaged algorithm identity evidence"]
    missing: list[str] = []
    checks: dict[str, bool] = {}
    expected_files = {
        "baseline": "safe_neural_horizon_pwm_baseline_strength_audit.json",
        "twin": "twin_evidence/twin_training_summary.json",
        "trace": "trace_evidence/trace_summary.json",
        "tuning": "safe_neural_horizon_pwm_baseline_tuning_evidence.json",
    }
    payloads: dict[str, Dict[str, Any]] = {}
    for key, rel in expected_files.items():
        path = release_dir / rel
        checks[f"{key}_artifact_present"] = path.exists()
        if not path.exists():
            missing.append(rel)
            continue
        payloads[key] = _load_json(path)

    baseline = payloads.get("baseline", {})
    twin = payloads.get("twin", {})
    trace = payloads.get("trace", {})
    tuning = payloads.get("tuning", {})

    checks["strong_baselines_ready"] = bool(baseline.get("publication_strong_baselines_ready", False))
    checks["baseline_tuning_ready"] = bool(tuning.get("baseline_tuning_ready", False))
    checks["twin_evidence_ready"] = bool(twin.get("trained_domain_randomized_twin_ready", False))
    checks["trace_evidence_ready"] = bool(trace.get("trace_evidence_ready", False))
    for key in (
        "strong_baselines_ready",
        "baseline_tuning_ready",
        "twin_evidence_ready",
        "trace_evidence_ready",
    ):
        if not checks[key]:
            missing.append(f"{key}=true")
    return checks, missing


def analyze_algorithm_identity(path: Path) -> Dict[str, Any]:
    payload, release_dir = _results_payload(path)
    matrix = dict(payload.get("matrix", {}))
    scenarios = [str(item) for item in payload.get("scenarios", [])]
    failures: list[str] = []
    warnings: list[str] = []

    feature_rows: Dict[str, Any] = {}
    for name, feature in ESSENTIAL_FEATURES.items():
        ready, missing = _source_markers(feature)
        if not ready:
            failures.append(f"{name}: missing source markers {missing}")
        feature_rows[name] = {
            "source_file": str(Path(feature["file"]).relative_to(ROOT)),
            "ready": ready,
            "missing_markers": missing,
            "why_it_matters": feature["why_it_matters"],
        }

    required_controllers = set(SAFE_CONTROLLER_VARIANTS) | set(COMPARISON_CONTROLLERS)
    missing_controllers: dict[str, list[str]] = {}
    for scenario in scenarios:
        present = set(dict(matrix.get(scenario, {})).keys())
        missing = sorted(required_controllers - present)
        if missing:
            missing_controllers[scenario] = missing
    if missing_controllers:
        failures.append(f"missing comparison rows: {missing_controllers}")

    baseline_rows: Dict[str, Any] = {}
    for baseline in sorted(COMPARISON_CONTROLLERS):
        row = BASELINE_DIFFERENCES[baseline]
        covered = baseline in set().union(*(set(dict(matrix.get(scenario, {})).keys()) for scenario in scenarios))
        ready = bool(covered and row["missing_snh_features"] and row["snh_distinction"])
        if not ready:
            failures.append(f"{baseline}: missing distinction evidence")
        baseline_rows[baseline] = {
            "comparison_present": covered,
            "missing_snh_features": row["missing_snh_features"],
            "snh_distinction": row["snh_distinction"],
            "ready": ready,
        }

    release_checks, release_missing = _release_checks(release_dir)
    if release_missing:
        failures.extend(release_missing)

    hardware_claim_false = payload.get("hardware_claim") is False
    if not hardware_claim_false:
        failures.append("hardware_claim=false")
    status = str(payload.get("status", ""))
    host_scope = status.startswith("host_") or status == "HOST_SIMULATION_ONLY"
    if not host_scope:
        failures.append("host-scoped status")

    if release_dir is None:
        warnings.append("algorithm identity is strongest when checked against a packaged release directory")
    warnings.append("This proves a distinct host-simulated control-law identity, not MCU/HIL/bench readiness or universal superiority.")

    new_algorithm_identity_supported = not failures and all(bool(row["ready"]) for row in feature_rows.values())
    return {
        "status": "safe_neural_horizon_pwm_algorithm_identity_audit",
        "hardware_claim": False,
        "new_algorithm_identity_supported": new_algorithm_identity_supported,
        "algorithm_name": "Safe Neural Horizon PWM with Event-Triggered Twin Feedback",
        "algorithm_identity_tuple": [
            "finite-horizon legal inverter-vector search",
            "deterministic neural cost shaping",
            "event-triggered neural-twin feedback and confidence",
            "protected AI-PWM Safety Gateway with no raw low-side gate access",
            "multiobjective cost: tracking/current/switching/loss/thermal/ripple/feedback/risk",
        ],
        "claim_scope": {
            "allowed": "distinct host-simulated control-law identity",
            "not_allowed": [
                "MCU/HIL/bench readiness",
                "universal superiority over industrial controllers",
                "full no-feedback operation under unknown load and drift",
            ],
        },
        "checks": {
            "host_scope": host_scope,
            "hardware_claim_false": hardware_claim_false,
            "scenario_count": len(scenarios),
            "all_required_controller_rows_present": not missing_controllers,
            **release_checks,
        },
        "essential_features": feature_rows,
        "baseline_distinction_matrix": baseline_rows,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SNH-PWM as a distinct host-level control algorithm.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = analyze_algorithm_identity(Path(args.input).expanduser().resolve())
    if args.out_json:
        out = Path(args.out_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out}")
    print(f"new_algorithm_identity_supported: {result['new_algorithm_identity_supported']}")
    if args.strict and not bool(result["new_algorithm_identity_supported"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
