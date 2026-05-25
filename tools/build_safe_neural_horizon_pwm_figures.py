from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _metric(row: Dict[str, Any], name: str, field: str = "mean") -> float:
    value = row.get(name, {})
    if isinstance(value, dict):
        return float(value.get(field, 0.0))
    return float(value or 0.0)


def _load(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        path = path / "safe_neural_horizon_pwm_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scenario, controllers in dict(payload.get("matrix", {})).items():
        for controller, raw in dict(controllers).items():
            if controller == "pareto_front":
                continue
            row = dict(raw)
            out.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "speed_error": _metric(row, "mean_abs_speed_error"),
                    "current": _metric(row, "mean_current_abs"),
                    "switches": _metric(row, "switch_events"),
                    "feedback": _metric(row, "feedback_usage_ratio"),
                    "fallback": _metric(row, "fallback_count"),
                    "failures": int(row.get("failure_count", 0)),
                    "safety_worst": _metric(row, "safety_violations", "worst"),
                }
            )
    return out


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return (start + end) / 2.0
    return start + (float(value) - min_value) * (end - start) / (max_value - min_value)


def _scatter_svg(rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, path: Path) -> None:
    width, height = 920, 620
    margin = 70
    x_vals = [float(row[x_key]) for row in rows]
    y_vals = [float(row[y_key]) for row in rows]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    colors = {
        "protected_ai_pwm_h1_baseline": "#8a8a8a",
        "safe_neural_horizon_pwm_h2": "#0b6bcb",
        "safe_neural_horizon_pwm_h3_thermal": "#12805c",
        "safe_neural_horizon_pwm_h4_sparse": "#b45f06",
        "fcs_mpc_one_step_baseline": "#6f42c1",
        "foc_svm_key_baseline": "#444444",
        "dtc_hysteresis_baseline": "#a23b72",
        "dtc_svm_baseline": "#d55e00",
        "deadbeat_current_baseline": "#007f7f",
        "sensorless_adaptive_foc_baseline": "#7f7f00",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Georgia,serif" font-size="24" fill="#1b1b1b">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Georgia,serif" font-size="15" fill="#333">{x_key}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Georgia,serif" font-size="15" fill="#333">{y_key}</text>',
    ]
    for row in rows:
        x = _scale(float(row[x_key]), x_min, x_max, margin, width - margin)
        y = _scale(float(row[y_key]), y_min, y_max, height - margin, margin)
        controller = str(row["controller"])
        color = colors.get(controller, "#999999")
        radius = 5.5 if controller.startswith("safe_neural") else 4.0
        opacity = "0.82" if controller.startswith("safe_neural") else "0.45"
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" opacity="{opacity}"/>')
    legend_y = 68
    for idx, (name, color) in enumerate(colors.items()):
        y = legend_y + idx * 22
        lines.append(f'<circle cx="{width-310}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="{width-296}" y="{y+5}" font-family="Georgia,serif" font-size="13" fill="#333">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bar_svg(rows: list[dict[str, Any]], path: Path) -> None:
    h2 = [row for row in rows if row["controller"] == "safe_neural_horizon_pwm_h2"]
    width = 1100
    bar_h = 18
    gap = 8
    margin_left = 230
    margin_right = 60
    height = 90 + len(h2) * (bar_h + gap)
    max_speed = max(float(row["speed_error"]) for row in h2) if h2 else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="550" y="34" text-anchor="middle" font-family="Georgia,serif" font-size="24" fill="#1b1b1b">SNH-PWM H2 scenario speed-error summary</text>',
    ]
    for idx, row in enumerate(h2):
        y = 66 + idx * (bar_h + gap)
        value = float(row["speed_error"])
        bar_w = _scale(value, 0.0, max_speed, 0.0, width - margin_left - margin_right)
        color = "#0b6bcb" if int(row["failures"]) == 0 else "#b00020"
        lines.append(f'<text x="{margin_left-12}" y="{y+14}" text-anchor="end" font-family="Georgia,serif" font-size="12" fill="#333">{row["scenario"]}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.82"/>')
        lines.append(f'<text x="{margin_left + bar_w + 6:.2f}" y="{y+14}" font-family="Georgia,serif" font-size="12" fill="#333">{value:.2f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figures(input_path: Path, out_dir: Path) -> list[Path]:
    payload = _load(input_path)
    rows = _rows(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "safe_neural_horizon_pwm_summary.csv"
    _write_csv(csv_path, rows)
    scatter_speed_current = out_dir / "fig_speed_error_vs_current.svg"
    scatter_feedback_switch = out_dir / "fig_feedback_vs_switching.svg"
    h2_bars = out_dir / "fig_h2_scenario_speed_error.svg"
    _scatter_svg(rows, "speed_error", "current", "Speed Error vs Current Stress", scatter_speed_current)
    _scatter_svg(rows, "feedback", "switches", "Feedback Usage vs Switching", scatter_feedback_switch)
    _bar_svg(rows, h2_bars)
    return [csv_path, scatter_speed_current, scatter_feedback_switch, h2_bars]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate SVG/CSV figures for Safe Neural Horizon PWM results.")
    parser.add_argument("--input", required=True, help="Release directory or safe_neural_horizon_pwm_results.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    files = build_figures(Path(args.input).expanduser().resolve(), Path(args.out_dir).expanduser().resolve())
    for path in files:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
