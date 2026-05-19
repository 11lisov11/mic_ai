from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - depends on invocation path
    sys.path.insert(0, str(ROOT))

from tools.air56_unoq_hardware_acceptance import build_acceptance_summary


SCHEMA = "mic_theory.air56_unoq.hardware_acceptance.v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"log JSON must be an object: {path}")
    return payload


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "ok", "pass", "passed"}


def _bool(payload: dict[str, Any], *names: str, default: bool = False) -> bool:
    for name in names:
        if name in payload:
            return _truthy(payload.get(name), default=default)
    return bool(default)


def _float(payload: dict[str, Any], *names: str, default: float = float("inf")) -> float:
    for name in names:
        if name not in payload:
            continue
        try:
            return float(payload.get(name))
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _csv_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _max_delta_ms_from_values(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    ordered = sorted(values)
    return max(b - a for a, b in zip(ordered, ordered[1:]))


def _max_period_ms_from_frames(frames: Any) -> float:
    if not isinstance(frames, list):
        return float("inf")
    times = []
    for frame in frames:
        if isinstance(frame, dict):
            value = frame.get("telemetry_t_ms", frame.get("t_ms"))
            try:
                times.append(float(value))
            except (TypeError, ValueError):
                continue
    return _max_delta_ms_from_values(times)


def _max_period_ms_from_csv(rows: list[dict[str, str]]) -> float:
    times = []
    for row in rows:
        value = _csv_float(row, "t_ms", "time_ms", "timestamp_ms")
        if value is not None:
            times.append(value)
    return _max_delta_ms_from_values(times)


def _decoded_mismatch_pct(rows: list[dict[str, str]], *, abs_tol: float = 1e-3, rel_tol: float = 0.01) -> float:
    pairs = [
        ("omega_meas", "stm_omega_meas"),
        ("omega_ref", "stm_omega_ref"),
        ("id", "stm_id"),
        ("i_d", "stm_id"),
        ("iq", "stm_iq"),
        ("i_q", "stm_iq"),
        ("vdc", "stm_vdc"),
        ("v_dc", "stm_vdc"),
        ("i_rms", "stm_i_rms"),
        ("irms", "stm_i_rms"),
        ("p_in", "stm_p_in"),
        ("pin", "stm_p_in"),
    ]
    total = 0
    mismatched = 0
    for row in rows:
        for linux_name, stm_name in pairs:
            linux_value = _csv_float(row, linux_name)
            stm_value = _csv_float(row, stm_name)
            if linux_value is None or stm_value is None:
                continue
            total += 1
            tolerance = max(abs_tol, rel_tol * max(abs(stm_value), 1.0))
            if abs(linux_value - stm_value) > tolerance:
                mismatched += 1
    if total == 0:
        return 100.0
    return 100.0 * float(mismatched) / float(total)


def _stage0_from_log(log: dict[str, Any]) -> dict[str, Any]:
    period = _float(log, "telemetry_period_ms_max")
    if not math.isfinite(period):
        period = _max_period_ms_from_frames(log.get("frames"))
    fallback_ms = _float(log, "fallback_ms")
    if not math.isfinite(fallback_ms) and _bool(log, "fallback_after_timeout"):
        fallback_ms = _float(log, "timeout_ms")
    duration_s = _float(log, "loopback_duration_s", "duration_s", default=0.0)
    struct_sizes_ok = _bool(log, "struct_sizes_ok") or (
        int(_float(log, "telemetry_size", default=0.0)) == 20 and int(_float(log, "command_size", default=0.0)) == 9
    )
    crc_error_rejected = _bool(log, "crc_error_rejected")
    passed = (
        _bool(log, "passed", default=True)
        and struct_sizes_ok
        and crc_error_rejected
        and duration_s >= 600.0
        and fallback_ms <= 100.0
        and period <= 12.0
    )
    return {
        "passed": passed,
        "struct_sizes_ok": struct_sizes_ok,
        "crc_error_rejected": crc_error_rejected,
        "loopback_duration_s": duration_s,
        "fallback_ms": fallback_ms,
        "telemetry_period_ms_max": period,
    }


def _stage1_from_log(log: dict[str, Any]) -> dict[str, Any]:
    result = {
        "mock_adapter_enabled": _bool(log, "mock_adapter_enabled"),
        "production_build_without_mock": _bool(log, "production_build_without_mock"),
        "current_scaling_ok": _bool(log, "current_scaling_ok"),
        "speed_scaling_ok": _bool(log, "speed_scaling_ok"),
        "vdc_scaling_ok": _bool(log, "vdc_scaling_ok"),
        "p_in_estimate_ok": _bool(log, "p_in_estimate_ok", "pin_estimate_ok"),
        "fault_bits_ok": _bool(log, "fault_bits_ok"),
        "safe_disable_ok": _bool(log, "safe_disable_ok"),
    }
    result["passed"] = (
        _bool(log, "passed", default=True)
        and not result["mock_adapter_enabled"]
        and all(bool(value) for key, value in result.items() if key != "mock_adapter_enabled")
    )
    return result


def _stage2_from_logs(log: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    period = _float(log, "telemetry_period_ms_max")
    if rows and not math.isfinite(period):
        period = _max_period_ms_from_csv(rows)
    mismatch = _float(log, "decoded_telemetry_mismatch_pct")
    if rows and not math.isfinite(mismatch):
        mismatch = _decoded_mismatch_pct(rows)
    ai_enabled = _bool(log, "ai_enabled", default=True)
    bridge_dry_run = _bool(log, "bridge_dry_run", "dry_run", "telemetry_only")
    passed = _bool(log, "passed", default=True) and not ai_enabled and bridge_dry_run and period <= 12.0 and mismatch <= 2.0
    return {
        "passed": passed,
        "ai_enabled": ai_enabled,
        "bridge_dry_run": bridge_dry_run,
        "telemetry_period_ms_max": period,
        "decoded_telemetry_mismatch_pct": mismatch,
    }


def _stage3_from_log(log: dict[str, Any]) -> dict[str, Any]:
    ai_enabled = _bool(log, "ai_enabled")
    tight_limits = _bool(log, "id_ref_limits_tight", "tight_limits")
    disable_on_fault = _bool(log, "disable_on_fault")
    fallback_ms = _float(log, "fallback_ms")
    tracking_regression = _bool(log, "tracking_guard_regression")
    passed = (
        _bool(log, "passed", default=True)
        and ai_enabled
        and tight_limits
        and disable_on_fault
        and fallback_ms <= 100.0
        and not tracking_regression
    )
    return {
        "passed": passed,
        "ai_enabled": ai_enabled,
        "id_ref_limits_tight": tight_limits,
        "disable_on_fault": disable_on_fault,
        "fallback_ms": fallback_ms,
        "tracking_guard_regression": tracking_regression,
    }


def _stage4_from_log(log: dict[str, Any]) -> dict[str, Any]:
    documented = _bool(log, "documented")
    guard_fail_delta = _float(log, "guard_fail_delta", default=1.0)
    tracking_regression = _bool(log, "tracking_guard_regression")
    thermal_ok = _bool(log, "current_thermal_limit_ok")
    fallback_oscillation = _bool(log, "fallback_oscillation")
    power_saving_pct = _float(log, "power_saving_pct", default=-1.0)
    passed = (
        _bool(log, "passed", default=True)
        and documented
        and guard_fail_delta <= 0.0
        and not tracking_regression
        and thermal_ok
        and not fallback_oscillation
        and power_saving_pct >= 0.0
    )
    return {
        "passed": passed,
        "documented": documented,
        "guard_fail_delta": guard_fail_delta,
        "tracking_guard_regression": tracking_regression,
        "current_thermal_limit_ok": thermal_ok,
        "fallback_oscillation": fallback_oscillation,
        "power_saving_pct": power_saving_pct,
    }


def build_hardware_report(
    *,
    board_id: str,
    operator: str,
    stage0: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage2_rows: list[dict[str, str]],
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "board_id": str(board_id).strip(),
        "operator": str(operator).strip(),
        "notes": str(notes).strip(),
        "stages": {
            "stage0": _stage0_from_log(stage0),
            "stage1": _stage1_from_log(stage1),
            "stage2": _stage2_from_logs(stage2, stage2_rows),
            "stage3": _stage3_from_log(stage3),
            "stage4": _stage4_from_log(stage4),
        },
    }


def build_hardware_report_from_paths(
    *,
    board_id: str,
    operator: str,
    stage0_json: Path,
    stage1_json: Path,
    stage2_json: Path,
    stage2_csv: Path | None,
    stage3_json: Path,
    stage4_json: Path,
    notes: str = "",
) -> dict[str, Any]:
    return build_hardware_report(
        board_id=board_id,
        operator=operator,
        stage0=_read_json(stage0_json),
        stage1=_read_json(stage1_json),
        stage2=_read_json(stage2_json),
        stage2_rows=_read_csv_rows(stage2_csv),
        stage3=_read_json(stage3_json),
        stage4=_read_json(stage4_json),
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AIR56 UNO Q hardware acceptance report from Stage 0-4 logs.")
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--stage0-json", required=True)
    parser.add_argument("--stage1-json", required=True)
    parser.add_argument("--stage2-json", required=True)
    parser.add_argument("--stage2-csv", default="")
    parser.add_argument("--stage3-json", required=True)
    parser.add_argument("--stage4-json", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-summary-json", default="")
    args = parser.parse_args()

    stage2_csv = Path(str(args.stage2_csv)).resolve() if str(args.stage2_csv).strip() else None
    report = build_hardware_report_from_paths(
        board_id=str(args.board_id),
        operator=str(args.operator),
        stage0_json=Path(str(args.stage0_json)).resolve(),
        stage1_json=Path(str(args.stage1_json)).resolve(),
        stage2_json=Path(str(args.stage2_json)).resolve(),
        stage2_csv=stage2_csv,
        stage3_json=Path(str(args.stage3_json)).resolve(),
        stage4_json=Path(str(args.stage4_json)).resolve(),
        notes=str(args.notes),
    )
    out_json = Path(str(args.out_json)).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = build_acceptance_summary(report)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if str(args.out_summary_json).strip():
        summary_path = Path(str(args.out_summary_json)).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0 if bool(summary["hardware_ready"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
