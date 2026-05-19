from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "mic_theory.air56_unoq.stage4_ab.v1"
REQUIRED_COLUMNS = ("t_ms", "omega_meas", "omega_ref", "p_in", "i_rms", "guard_fail", "fallback_event")


@dataclass(frozen=True)
class RunMetrics:
    name: str
    samples: int
    duration_s: float
    mean_p_in_w: float
    tracking_mae_rad_s: float
    guard_fail_count: int
    max_i_rms_a: float
    fallback_event_count: int
    fallback_transition_count: int
    thermal_fault_count: int


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV log is empty: {path}")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"CSV log {path} misses required columns: {', '.join(missing)}")
    return rows


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {name}: {row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value for {name}: {row.get(name)!r}")
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "fault", "fail", "failed"}


def _bool(row: dict[str, str], name: str) -> bool:
    return _truthy(row.get(name, "0"))


def _duration_s(times_ms: list[float]) -> float:
    if len(times_ms) < 2:
        return 0.0
    return max(0.0, (max(times_ms) - min(times_ms)) / 1000.0)


def _fallback_transitions(values: list[bool]) -> int:
    if len(values) < 2:
        return 0
    return sum(1 for prev, cur in zip(values, values[1:]) if prev != cur)


def analyze_run(name: str, csv_path: Path) -> RunMetrics:
    rows = _read_csv_rows(csv_path)
    times = [_float(row, "t_ms") for row in rows]
    p_in = [_float(row, "p_in") for row in rows]
    i_rms = [_float(row, "i_rms") for row in rows]
    tracking_abs = [abs(_float(row, "omega_meas") - _float(row, "omega_ref")) for row in rows]
    guard_flags = [_bool(row, "guard_fail") for row in rows]
    fallback_flags = [_bool(row, "fallback_event") for row in rows]
    thermal_flags = [_bool(row, "thermal_fault") for row in rows] if "thermal_fault" in rows[0] else [False for _ in rows]

    if len(rows) < 2:
        raise ValueError(f"CSV log must contain at least two samples: {csv_path}")
    mean_p_in = sum(p_in) / float(len(p_in))
    if mean_p_in <= 0.0:
        raise ValueError(f"mean p_in must be positive for {csv_path}")

    return RunMetrics(
        name=str(name),
        samples=len(rows),
        duration_s=round(_duration_s(times), 6),
        mean_p_in_w=mean_p_in,
        tracking_mae_rad_s=sum(tracking_abs) / float(len(tracking_abs)),
        guard_fail_count=sum(1 for item in guard_flags if item),
        max_i_rms_a=max(i_rms),
        fallback_event_count=sum(1 for item in fallback_flags if item),
        fallback_transition_count=_fallback_transitions(fallback_flags),
        thermal_fault_count=sum(1 for item in thermal_flags if item),
    )


def _weighted_mean(metrics: list[RunMetrics], field: str) -> float:
    total_samples = sum(item.samples for item in metrics)
    if total_samples <= 0:
        return float("inf")
    return sum(float(getattr(item, field)) * item.samples for item in metrics) / float(total_samples)


def _sum(metrics: list[RunMetrics], field: str) -> int:
    return sum(int(getattr(item, field)) for item in metrics)


def build_stage4_summary(
    *,
    foc_no_load: RunMetrics,
    foc_load_step: RunMetrics,
    ai_no_load: RunMetrics,
    ai_load_step: RunMetrics,
    max_current_rms_a: float,
    tracking_abs_tol_rad_s: float = 0.1,
    tracking_rel_tol: float = 0.02,
    min_power_saving_pct: float = 0.0,
    max_fallback_events: int = 0,
    max_fallback_transitions: int = 0,
) -> dict[str, Any]:
    foc_runs = [foc_no_load, foc_load_step]
    ai_runs = [ai_no_load, ai_load_step]
    foc_power = _weighted_mean(foc_runs, "mean_p_in_w")
    ai_power = _weighted_mean(ai_runs, "mean_p_in_w")
    power_saving_pct = 100.0 * (foc_power - ai_power) / foc_power

    foc_tracking = _weighted_mean(foc_runs, "tracking_mae_rad_s")
    ai_tracking = _weighted_mean(ai_runs, "tracking_mae_rad_s")
    tracking_allowed = foc_tracking + max(float(tracking_abs_tol_rad_s), abs(foc_tracking) * float(tracking_rel_tol))
    tracking_guard_regression = ai_tracking > tracking_allowed

    foc_guard_fail = _sum(foc_runs, "guard_fail_count")
    ai_guard_fail = _sum(ai_runs, "guard_fail_count")
    guard_fail_delta = ai_guard_fail - foc_guard_fail

    ai_max_current = max(item.max_i_rms_a for item in ai_runs)
    ai_thermal_faults = _sum(ai_runs, "thermal_fault_count")
    current_thermal_limit_ok = ai_max_current <= float(max_current_rms_a) and ai_thermal_faults == 0

    ai_fallback_events = _sum(ai_runs, "fallback_event_count")
    ai_fallback_transitions = _sum(ai_runs, "fallback_transition_count")
    fallback_oscillation = ai_fallback_events > int(max_fallback_events) or ai_fallback_transitions > int(max_fallback_transitions)

    documented = all(item.samples >= 2 and item.duration_s > 0.0 for item in [*foc_runs, *ai_runs])
    passed = (
        documented
        and guard_fail_delta <= 0
        and not tracking_guard_regression
        and current_thermal_limit_ok
        and not fallback_oscillation
        and power_saving_pct >= float(min_power_saving_pct)
    )

    return {
        "schema": SCHEMA,
        "passed": passed,
        "documented": documented,
        "guard_fail_delta": guard_fail_delta,
        "tracking_guard_regression": tracking_guard_regression,
        "current_thermal_limit_ok": current_thermal_limit_ok,
        "fallback_oscillation": fallback_oscillation,
        "power_saving_pct": power_saving_pct,
        "thresholds": {
            "max_current_rms_a": float(max_current_rms_a),
            "tracking_abs_tol_rad_s": float(tracking_abs_tol_rad_s),
            "tracking_rel_tol": float(tracking_rel_tol),
            "min_power_saving_pct": float(min_power_saving_pct),
            "max_fallback_events": int(max_fallback_events),
            "max_fallback_transitions": int(max_fallback_transitions),
        },
        "metrics": {
            "foc_mean_p_in_w": foc_power,
            "ai_mean_p_in_w": ai_power,
            "foc_tracking_mae_rad_s": foc_tracking,
            "ai_tracking_mae_rad_s": ai_tracking,
            "foc_guard_fail_count": foc_guard_fail,
            "ai_guard_fail_count": ai_guard_fail,
            "ai_max_i_rms_a": ai_max_current,
            "ai_thermal_fault_count": ai_thermal_faults,
            "ai_fallback_event_count": ai_fallback_events,
            "ai_fallback_transition_count": ai_fallback_transitions,
            "runs": {
                "foc_no_load": asdict(foc_no_load),
                "foc_load_step": asdict(foc_load_step),
                "ai_no_load": asdict(ai_no_load),
                "ai_load_step": asdict(ai_load_step),
            },
        },
    }


def build_stage4_summary_from_paths(
    *,
    foc_no_load_csv: Path,
    foc_load_step_csv: Path,
    ai_no_load_csv: Path,
    ai_load_step_csv: Path,
    max_current_rms_a: float,
    tracking_abs_tol_rad_s: float = 0.1,
    tracking_rel_tol: float = 0.02,
    min_power_saving_pct: float = 0.0,
    max_fallback_events: int = 0,
    max_fallback_transitions: int = 0,
) -> dict[str, Any]:
    return build_stage4_summary(
        foc_no_load=analyze_run("foc_no_load", foc_no_load_csv),
        foc_load_step=analyze_run("foc_load_step", foc_load_step_csv),
        ai_no_load=analyze_run("ai_no_load", ai_no_load_csv),
        ai_load_step=analyze_run("ai_load_step", ai_load_step_csv),
        max_current_rms_a=max_current_rms_a,
        tracking_abs_tol_rad_s=tracking_abs_tol_rad_s,
        tracking_rel_tol=tracking_rel_tol,
        min_power_saving_pct=min_power_saving_pct,
        max_fallback_events=max_fallback_events,
        max_fallback_transitions=max_fallback_transitions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze physical AIR56 Stage 4 FOC baseline vs MIC/AI A/B CSV logs.")
    parser.add_argument("--foc-no-load-csv", required=True)
    parser.add_argument("--foc-load-step-csv", required=True)
    parser.add_argument("--ai-no-load-csv", required=True)
    parser.add_argument("--ai-load-step-csv", required=True)
    parser.add_argument("--max-current-rms-a", type=float, required=True)
    parser.add_argument("--tracking-abs-tol-rad-s", type=float, default=0.1)
    parser.add_argument("--tracking-rel-tol", type=float, default=0.02)
    parser.add_argument("--min-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--max-fallback-events", type=int, default=0)
    parser.add_argument("--max-fallback-transitions", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    summary = build_stage4_summary_from_paths(
        foc_no_load_csv=Path(str(args.foc_no_load_csv)).resolve(),
        foc_load_step_csv=Path(str(args.foc_load_step_csv)).resolve(),
        ai_no_load_csv=Path(str(args.ai_no_load_csv)).resolve(),
        ai_load_step_csv=Path(str(args.ai_load_step_csv)).resolve(),
        max_current_rms_a=float(args.max_current_rms_a),
        tracking_abs_tol_rad_s=float(args.tracking_abs_tol_rad_s),
        tracking_rel_tol=float(args.tracking_rel_tol),
        min_power_saving_pct=float(args.min_power_saving_pct),
        max_fallback_events=int(args.max_fallback_events),
        max_fallback_transitions=int(args.max_fallback_transitions),
    )
    out_json = Path(str(args.out_json)).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
