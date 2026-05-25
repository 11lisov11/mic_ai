from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _metric(row: Dict[str, Any], name: str) -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> List[str]:
    headers = [str(h) for h in headers]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def _controller_rows(payload: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    controllers = dict(payload.get("controllers", {}))
    for name, raw in controllers.items():
        row = dict(raw)
        rows.append(
            [
                name,
                _fmt(_metric(row, "mean_abs_speed_error")),
                _fmt(_metric(row, "mean_current_abs")),
                _fmt(_metric(row, "switch_events")),
                _fmt(_metric(row, "feedback_usage_ratio")),
                _fmt(_metric(row, "fallback_count")),
                str(row.get("failure_count", 0)),
            ]
        )
    return rows


def _matrix_rows(scenario_payload: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for name, raw in scenario_payload.items():
        if name == "pareto_front":
            continue
        row = dict(raw)
        rows.append(
            [
                name,
                _fmt(_metric(row, "mean_abs_speed_error")),
                _fmt(_metric(row, "mean_current_abs")),
                _fmt(_metric(row, "switch_events")),
                _fmt(_metric(row, "feedback_usage_ratio")),
                _fmt(_metric(row, "fallback_count")),
                str(row.get("failure_count", 0)),
            ]
        )
    return rows


def build_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Safe Neural Horizon PWM Host Research Report")
    lines.append("")
    lines.append(f"- status: `{payload.get('status', 'unknown')}`")
    lines.append(f"- hardware_claim: `{bool(payload.get('hardware_claim', False))}`")
    lines.append(f"- mc_trials: `{payload.get('mc_trials', 0)}`")
    lines.append(f"- steps_per_trial: `{payload.get('steps_per_trial', 0)}`")
    lines.append(f"- seed: `{payload.get('seed', '')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is a host-level simulation report. It is not MCU, HIL, or bench evidence.")
    lines.append("The comparison matrix uses named host baselines; none of those rows is hardware, HIL, or publication-tuned evidence.")
    lines.append("")

    if "controllers" in payload:
        lines.append("## Controller Summary")
        lines.extend(
            _table(
                ["controller", "speed_err", "current", "switches", "feedback", "fallback", "failures"],
                _controller_rows(payload),
            )
        )
        lines.append("")
        lines.append("Pareto front:")
        for item in payload.get("pareto_front", []):
            lines.append(f"- `{item}`")
        lines.append("")

    matrix = dict(payload.get("matrix", {}))
    if matrix:
        lines.append("## Scenario Matrix")
        lines.append("")
        for scenario, raw in matrix.items():
            scenario_payload = dict(raw)
            lines.append(f"### {scenario}")
            lines.extend(
                _table(
                    ["controller", "speed_err", "current", "switches", "feedback", "fallback", "failures"],
                    _matrix_rows(scenario_payload),
                )
            )
            lines.append("")
            lines.append("Pareto front:")
            for item in scenario_payload.get("pareto_front", []):
                lines.append(f"- `{item}`")
            lines.append("")

    ablation = dict(payload.get("ablation", {}))
    if ablation:
        lines.append("## Ablation")
        lines.extend(
            _table(
                ["variant", "speed_err", "current", "switches", "feedback", "fallback", "failures"],
                _matrix_rows(ablation),
            )
        )
        lines.append("")
        lines.append("Ablation Pareto front:")
        for item in ablation.get("pareto_front", []):
            lines.append(f"- `{item}`")
        lines.append("")

    fault = dict(payload.get("fault_injection", {}))
    if fault:
        lines.append("## Fault Injection")
        lines.append("")
        lines.append(f"- all_gateway_cases_no_shoot_through: `{bool(fault.get('all_gateway_cases_no_shoot_through', False))}`")
        lines.append(f"- raw_shoot_through_detector_triggered: `{bool(fault.get('raw_shoot_through_detector_triggered', False))}`")
        rows = []
        for name, raw in dict(fault.get("cases", {})).items():
            case = dict(raw)
            rows.append(
                [
                    name,
                    str(bool(case.get("accepted", False))),
                    str(bool(case.get("pwm_enabled", False))),
                    str(case.get("fault_flags", 0)),
                    str(bool(case.get("fault_latched", False))),
                    str(bool(case.get("shoot_through", False))),
                ]
            )
        lines.extend(_table(["case", "accepted", "pwm_enabled", "fault_flags", "latched", "shoot_through"], rows))
        lines.append("")

    lines.append("## Honest Status")
    lines.append("")
    lines.append("- Shown: host-level vector safety, scenario smoke, ablation smoke, Pareto extraction.")
    lines.append("- Not shown: publication-tuned FOC-SVM/DTC-SVM/deadbeat strength, trained neural twin, MCU timing, HIL, or bench safety.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown report from Safe Neural Horizon PWM JSON results.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    report = build_report(payload)
    out = Path(args.out_md).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
