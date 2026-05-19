from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_STAGES = ("stage0", "stage1", "stage2", "stage3", "stage4")


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"hardware acceptance report must be a JSON object: {path}")
    return payload


def _stage(payload: dict[str, Any], name: str) -> dict[str, Any]:
    stages = payload.get("stages", {})
    if not isinstance(stages, dict):
        return {}
    value = stages.get(name, {})
    return value if isinstance(value, dict) else {}


def _bool(stage: dict[str, Any], field: str) -> bool:
    return bool(stage.get(field, False))


def _float(stage: dict[str, Any], field: str, default: float = float("inf")) -> float:
    try:
        return float(stage.get(field, default))
    except Exception:
        return float(default)


def evaluate_hardware_acceptance(payload: dict[str, Any]) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []
    checks.append(
        AcceptanceCheck(
            "schema",
            str(payload.get("schema", "")) == "mic_theory.air56_unoq.hardware_acceptance.v1",
            "report schema must be mic_theory.air56_unoq.hardware_acceptance.v1",
        )
    )
    checks.append(AcceptanceCheck("board_id", bool(str(payload.get("board_id", "")).strip()), "board_id is required"))
    checks.append(AcceptanceCheck("operator", bool(str(payload.get("operator", "")).strip()), "operator is required"))

    for name in REQUIRED_STAGES:
        stage = _stage(payload, name)
        checks.append(AcceptanceCheck(f"{name}.present", bool(stage), f"{name} report is required"))
        checks.append(AcceptanceCheck(f"{name}.passed", _bool(stage, "passed"), f"{name} must pass"))

    stage0 = _stage(payload, "stage0")
    checks.extend(
        [
            AcceptanceCheck("stage0.struct_sizes", _bool(stage0, "struct_sizes_ok"), "telemetry=20 and command=9 bytes"),
            AcceptanceCheck("stage0.crc", _bool(stage0, "crc_error_rejected"), "CRC failures must be rejected"),
            AcceptanceCheck("stage0.duration", _float(stage0, "loopback_duration_s", 0.0) >= 600.0, "loopback duration must be >= 600 s"),
            AcceptanceCheck("stage0.fallback", _float(stage0, "fallback_ms") <= 100.0, "fallback must be <= 100 ms"),
            AcceptanceCheck("stage0.period", _float(stage0, "telemetry_period_ms_max") <= 12.0, "10 ms link must stay within 12 ms max"),
        ]
    )

    stage1 = _stage(payload, "stage1")
    checks.extend(
        [
            AcceptanceCheck("stage1.production_no_mock", not _bool(stage1, "mock_adapter_enabled"), "mock adapter must be disabled"),
            AcceptanceCheck("stage1.production_build", _bool(stage1, "production_build_without_mock"), "production build without mock must pass"),
            AcceptanceCheck("stage1.current_scaling", _bool(stage1, "current_scaling_ok"), "current scaling must be validated"),
            AcceptanceCheck("stage1.speed_scaling", _bool(stage1, "speed_scaling_ok"), "speed scaling must be validated"),
            AcceptanceCheck("stage1.vdc_scaling", _bool(stage1, "vdc_scaling_ok"), "Vdc scaling must be validated"),
            AcceptanceCheck("stage1.pin_estimate", _bool(stage1, "p_in_estimate_ok"), "P_in estimate must be validated"),
            AcceptanceCheck("stage1.fault_bits", _bool(stage1, "fault_bits_ok"), "fault bits must be validated"),
            AcceptanceCheck("stage1.safe_disable", _bool(stage1, "safe_disable_ok"), "safe disable path must be validated"),
        ]
    )

    stage2 = _stage(payload, "stage2")
    checks.extend(
        [
            AcceptanceCheck("stage2.ai_disabled", not _bool(stage2, "ai_enabled"), "AI must be disabled in telemetry-only stage"),
            AcceptanceCheck("stage2.dry_run", _bool(stage2, "bridge_dry_run"), "bridge must run dry-run/telemetry-only"),
            AcceptanceCheck("stage2.period", _float(stage2, "telemetry_period_ms_max") <= 12.0, "telemetry period max must be <= 12 ms"),
            AcceptanceCheck("stage2.decode_match", _float(stage2, "decoded_telemetry_mismatch_pct") <= 2.0, "decoded telemetry mismatch must be <= 2%"),
        ]
    )

    stage3 = _stage(payload, "stage3")
    checks.extend(
        [
            AcceptanceCheck("stage3.ai_enabled", _bool(stage3, "ai_enabled"), "AI must be enabled in tight-limit stage"),
            AcceptanceCheck("stage3.tight_limits", _bool(stage3, "id_ref_limits_tight"), "id_ref limits must be tight"),
            AcceptanceCheck("stage3.disable_on_fault", _bool(stage3, "disable_on_fault"), "disable-on-fault must be enabled"),
            AcceptanceCheck("stage3.fallback", _float(stage3, "fallback_ms") <= 100.0, "fallback must be <= 100 ms"),
            AcceptanceCheck("stage3.no_tracking_regression", not _bool(stage3, "tracking_guard_regression"), "tracking guard must not regress"),
        ]
    )

    stage4 = _stage(payload, "stage4")
    checks.extend(
        [
            AcceptanceCheck("stage4.documented", _bool(stage4, "documented"), "A/B result must be documented"),
            AcceptanceCheck("stage4.no_guard_regression", _float(stage4, "guard_fail_delta", 1.0) <= 0.0, "guard failures must not increase"),
            AcceptanceCheck("stage4.no_tracking_regression", not _bool(stage4, "tracking_guard_regression"), "tracking must not regress"),
            AcceptanceCheck("stage4.thermal", _bool(stage4, "current_thermal_limit_ok"), "current/thermal limit must pass"),
            AcceptanceCheck("stage4.no_fallback_oscillation", not _bool(stage4, "fallback_oscillation"), "fallback must not oscillate"),
            AcceptanceCheck("stage4.power_nonnegative", _float(stage4, "power_saving_pct", -1.0) >= 0.0, "AI power saving must be non-negative"),
        ]
    )
    return checks


def build_acceptance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    checks = evaluate_hardware_acceptance(payload)
    return {
        "hardware_ready": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AIR56 UNO Q physical hardware acceptance report.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    report_path = Path(str(args.report)).resolve()
    payload = _load_report(report_path)
    summary = build_acceptance_summary(payload)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(summary["hardware_ready"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
