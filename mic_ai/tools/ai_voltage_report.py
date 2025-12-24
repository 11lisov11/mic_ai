from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_ai_voltage_report(
    motor_runs: Any,
    eval_runs: Any = None,
    distillation: Any = None,
    report_path: Path | None = None,
) -> Path:
    if report_path is None:
        report_path = Path("outputs/demo_ai/final_report.json")
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "motor_runs": motor_runs,
        "eval_runs": eval_runs,
        "distillation": distillation,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def print_summary(report_path: Path) -> None:
    if not report_path.exists():
        print(f"[report] missing: {report_path}")
        return
    data = json.loads(report_path.read_text(encoding="utf-8"))
    motor_runs = data.get("motor_runs")
    eval_runs = data.get("eval_runs")
    print(f"[report] saved: {report_path}")
    if motor_runs is not None:
        print(f"[report] motor_runs: {len(motor_runs) if hasattr(motor_runs, '__len__') else 'n/a'}")
    if eval_runs is not None:
        print(f"[report] eval_runs: {len(eval_runs) if hasattr(eval_runs, '__len__') else 'n/a'}")
