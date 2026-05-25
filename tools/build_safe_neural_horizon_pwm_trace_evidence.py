from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.induction_motor_alpha_beta import AlphaBetaInductionMotorModel, AlphaBetaMotorParams, AlphaBetaMotorState, randomized_motor_params
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage, switch_events
from safety.ai_pwm_gateway import has_shoot_through, transition_waveform
from tools.run_safe_neural_horizon_pwm_study import _controller, _controller_specs, _make_base_params, _scenario_values


DEFAULT_TRACE_CONTROLLERS = [
    "protected_ai_pwm_h1_baseline",
    "fcs_mpc_one_step_baseline",
    "foc_svm_key_baseline",
    "dtc_hysteresis_baseline",
    "dtc_svm_baseline",
    "deadbeat_current_baseline",
    "sensorless_adaptive_foc_baseline",
    "safe_neural_horizon_pwm_h2",
]


def _finite(value: float, fallback: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _rms(values: Iterable[float]) -> float:
    arr = np.asarray([_finite(v) for v in values], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _fft_metrics(values: Iterable[float], sample_period_s: float) -> Dict[str, float]:
    arr = np.asarray([_finite(v) for v in values], dtype=float)
    if arr.size < 8:
        return {
            "dc": float(np.mean(arr)) if arr.size else 0.0,
            "dominant_freq_hz": 0.0,
            "dominant_amp": 0.0,
            "thd_like": 0.0,
            "high_freq_energy_ratio": 0.0,
        }
    centered = arr - float(np.mean(arr))
    amps = np.abs(np.fft.rfft(centered)) * (2.0 / float(arr.size))
    freqs = np.fft.rfftfreq(arr.size, d=max(float(sample_period_s), 1.0e-12))
    if amps.size <= 1:
        dominant_idx = 0
    else:
        dominant_idx = int(np.argmax(amps[1:]) + 1)
    dominant_amp = float(amps[dominant_idx]) if dominant_idx > 0 else 0.0
    harmonic_energy = 0.0
    for harmonic in range(2, 8):
        idx = dominant_idx * harmonic
        if 0 < idx < amps.size:
            harmonic_energy += float(amps[idx]) ** 2
    signal_energy = float(np.sum(np.square(amps[1:])))
    nyquist = 0.5 / max(float(sample_period_s), 1.0e-12)
    high_mask = freqs >= 0.25 * nyquist
    high_energy = float(np.sum(np.square(amps[high_mask])))
    return {
        "dc": float(np.mean(arr)),
        "dominant_freq_hz": float(freqs[dominant_idx]) if dominant_idx > 0 else 0.0,
        "dominant_amp": dominant_amp,
        "thd_like": float(math.sqrt(harmonic_energy) / max(dominant_amp, 1.0e-12)),
        "high_freq_energy_ratio": float(high_energy / max(signal_energy, 1.0e-12)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _controller_lookup(quick: bool = False) -> dict[str, tuple[int, int]]:
    return {label: (horizon, feedback_period) for label, horizon, feedback_period in _controller_specs(quick=quick)}


def _simulate_trace(
    *,
    label: str,
    horizon: int,
    feedback_period: int,
    scenario: str,
    steps: int,
    base_motor: AlphaBetaMotorParams,
    real_params: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    real_motor = AlphaBetaInductionMotorModel(real_params, AlphaBetaMotorState())
    controller = _controller(
        label=label,
        base_motor=base_motor,
        inverter=inverter,
        horizon=horizon,
        feedback_period=feedback_period,
    )
    controller.reset(AlphaBetaMotorState())

    omega_nom = 2.0 * math.pi * 50.0 / max(base_motor.p, 1)
    prev_vector = 0
    feedback_count = 0
    fallback_count = 0
    fault_latch_count = 0
    safety_violations = 0
    switch_total = 0
    measured_state_history: list[AlphaBetaMotorState] = []
    rows: list[dict[str, Any]] = []

    for k in range(max(int(steps), 1)):
        omega_ref, load_torque, vdc_scale, force_sensor_dropout = _scenario_values(
            scenario,
            k,
            steps,
            omega_nom,
        )
        step_inverter = replace(inverter, Vdc=float(inverter.Vdc) * float(vdc_scale))
        real_currents = real_motor.currents()
        measured_i_abs = real_currents.stator_abs
        measured_state_history.append(real_motor.state)
        measured_state = real_motor.state
        if scenario == "sensor_delay" and len(measured_state_history) > 6:
            measured_state = measured_state_history[-6]

        speed_error_pre = omega_ref - controller.twin.state_hat.omega_m
        use_feedback = (
            k == 0
            or k % max(feedback_period, 1) == 0
            or abs(speed_error_pre) > controller.cfg.feedback_error_threshold_rad_s
            or controller.twin.uncertainty > controller.cfg.feedback_uncertainty_threshold
        )
        if force_sensor_dropout and k > steps // 4:
            use_feedback = k % max(feedback_period * 6, 1) == 0
        if use_feedback:
            feedback_count += 1

        result = controller.step(
            omega_ref=omega_ref,
            load_torque_nm=load_torque,
            measured_state=measured_state if use_feedback else None,
            measured_i_abs=measured_i_abs,
            vdc=step_inverter.Vdc,
        )
        if not result.decision.accepted:
            fallback_count += 1
        if result.decision.fault_latched:
            fault_latch_count += 1

        waveform = transition_waveform(prev_vector, result.vector_id, dead_time_ticks=2)
        if has_shoot_through(waveform):
            safety_violations += 1
        switch_now = switch_events(prev_vector, result.vector_id)
        switch_total += switch_now
        prev_vector = result.vector_id

        if result.decision.pwm_enabled:
            v_alpha, v_beta = alpha_beta_voltage(
                result.vector_id,
                step_inverter,
                i_alpha_beta=(real_currents.i_s_alpha, real_currents.i_s_beta),
            )
        else:
            v_alpha, v_beta = 0.0, 0.0
        step = real_motor.step(v_alpha, v_beta, load_torque, step_inverter.t_pwm_s)
        rows.append(
            {
                "k": k,
                "t_s": k * step_inverter.t_pwm_s,
                "scenario": scenario,
                "controller": label,
                "omega_ref_rad_s": omega_ref,
                "omega_m_rad_s": step.state.omega_m,
                "speed_error_rad_s": omega_ref - step.state.omega_m,
                "load_torque_nm": load_torque,
                "i_s_alpha_a": step.currents.i_s_alpha,
                "i_s_beta_a": step.currents.i_s_beta,
                "i_abs_a": step.currents.stator_abs,
                "torque_nm": step.torque_nm,
                "vector_id": result.vector_id,
                "feedback_used": 1 if use_feedback else 0,
                "feedback_requested": 1 if result.feedback_requested else 0,
                "accepted": 1 if result.decision.accepted else 0,
                "pwm_enabled": 1 if result.decision.pwm_enabled else 0,
                "fallback": 0 if result.decision.accepted else 1,
                "fault_latched": 1 if result.decision.fault_latched else 0,
                "confidence": result.confidence,
                "predicted_risk": result.predicted_risk,
                "loss_w": float(result.metrics.get("loss_w", 0.0)),
                "tj_c": float(result.metrics.get("tj_c", 0.0)),
                "switch_events": switch_now,
                "vdc_v": step_inverter.Vdc,
            }
        )

    speed_errors = [float(row["speed_error_rad_s"]) for row in rows]
    currents = [float(row["i_abs_a"]) for row in rows]
    torques = [float(row["torque_nm"]) for row in rows]
    current_fft = _fft_metrics(currents, inverter.t_pwm_s)
    torque_fft = _fft_metrics(torques, inverter.t_pwm_s)
    metrics = {
        "mean_abs_speed_error": float(sum(abs(v) for v in speed_errors) / max(len(speed_errors), 1)),
        "speed_error_rms": _rms(speed_errors),
        "current_rms": _rms(currents),
        "current_max": float(max(currents) if currents else 0.0),
        "current_thd_like": float(current_fft["thd_like"]),
        "current_high_freq_energy_ratio": float(current_fft["high_freq_energy_ratio"]),
        "current_dominant_freq_hz": float(current_fft["dominant_freq_hz"]),
        "torque_rms": _rms(torques),
        "torque_thd_like": float(torque_fft["thd_like"]),
        "torque_high_freq_energy_ratio": float(torque_fft["high_freq_energy_ratio"]),
        "torque_dominant_freq_hz": float(torque_fft["dominant_freq_hz"]),
        "torque_ripple_rms": _rms([torques[i] - torques[i - 1] for i in range(1, len(torques))]),
        "switch_events": float(switch_total),
        "feedback_usage_ratio": float(feedback_count / max(steps, 1)),
        "fallback_count": float(fallback_count),
        "fault_latch_count": float(fault_latch_count),
        "safety_violations": float(safety_violations),
    }
    return rows, metrics


def _scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return (start + end) / 2.0
    return start + (float(value) - min_value) * (end - start) / (max_value - min_value)


def _speed_trace_svg(trace_rows: list[dict[str, Any]], scenario: str, path: Path) -> None:
    width, height = 1080, 620
    margin = 72
    by_controller: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        by_controller.setdefault(str(row["controller"]), []).append(row)
    times = [float(row["t_s"]) for row in trace_rows]
    speeds = [float(row["omega_m_rad_s"]) for row in trace_rows] + [float(row["omega_ref_rad_s"]) for row in trace_rows]
    x_min, x_max = min(times), max(times)
    y_min, y_max = min(speeds), max(speeds)
    colors = {
        "protected_ai_pwm_h1_baseline": "#777777",
        "fcs_mpc_one_step_baseline": "#6f42c1",
        "foc_svm_key_baseline": "#222222",
        "dtc_hysteresis_baseline": "#a23b72",
        "dtc_svm_baseline": "#d55e00",
        "deadbeat_current_baseline": "#007f7f",
        "sensorless_adaptive_foc_baseline": "#7f7f00",
        "safe_neural_horizon_pwm_h2": "#0b6bcb",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Georgia,serif" font-size="24" fill="#1b1b1b">Trace speed response: {scenario}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#333" stroke-width="1.5"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Georgia,serif" font-size="15">time, s</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Georgia,serif" font-size="15">omega, rad/s</text>',
    ]
    ref_rows = next(iter(by_controller.values()), [])
    ref_points = []
    for row in ref_rows:
        x = _scale(float(row["t_s"]), x_min, x_max, margin, width - margin)
        y = _scale(float(row["omega_ref_rad_s"]), y_min, y_max, height - margin, margin)
        ref_points.append(f"{x:.2f},{y:.2f}")
    if ref_points:
        lines.append(f'<polyline points="{" ".join(ref_points)}" fill="none" stroke="#111" stroke-width="2" stroke-dasharray="7 5" opacity="0.75"/>')
    for controller, rows in by_controller.items():
        points = []
        for row in rows:
            x = _scale(float(row["t_s"]), x_min, x_max, margin, width - margin)
            y = _scale(float(row["omega_m_rad_s"]), y_min, y_max, height - margin, margin)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors.get(controller, "#999")}" stroke-width="2" opacity="0.78"/>'
        )
    legend_y = 66
    lines.append(f'<line x1="{width-340}" y1="{legend_y}" x2="{width-310}" y2="{legend_y}" stroke="#111" stroke-width="2" stroke-dasharray="7 5"/>')
    lines.append(f'<text x="{width-300}" y="{legend_y+5}" font-family="Georgia,serif" font-size="13">omega_ref</text>')
    for idx, (controller, color) in enumerate(colors.items()):
        y = legend_y + 24 + idx * 21
        lines.append(f'<line x1="{width-340}" y1="{y}" x2="{width-310}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{width-300}" y="{y+5}" font-family="Georgia,serif" font-size="13">{controller}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spectral_bar_svg(summary_rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1040, 620
    margin_left = 290
    margin_right = 80
    bar_h = 14
    gap = 9
    max_value = max(
        [float(row["current_thd_like"]) for row in summary_rows]
        + [float(row["torque_thd_like"]) for row in summary_rows]
        + [1.0e-9]
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="520" y="34" text-anchor="middle" font-family="Georgia,serif" font-size="24" fill="#1b1b1b">Trace FFT/THD-like evidence</text>',
        '<text x="520" y="57" text-anchor="middle" font-family="Georgia,serif" font-size="13" fill="#555">host simulation, dominant-bin harmonic ratio; not hardware THD</text>',
    ]
    for idx, row in enumerate(summary_rows):
        y0 = 90 + idx * (2 * bar_h + gap + 14)
        label = str(row["controller"])
        lines.append(f'<text x="{margin_left-12}" y="{y0+12}" text-anchor="end" font-family="Georgia,serif" font-size="12" fill="#333">{label}</text>')
        for offset, key, color, name in [
            (0, "current_thd_like", "#0b6bcb", "I"),
            (bar_h + 3, "torque_thd_like", "#b45f06", "T"),
        ]:
            value = float(row[key])
            bar_w = _scale(value, 0.0, max_value, 0.0, width - margin_left - margin_right)
            y = y0 + offset
            lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.82"/>')
            lines.append(f'<text x="{margin_left + bar_w + 6:.2f}" y="{y+11}" font-family="Georgia,serif" font-size="11" fill="#333">{name}: {value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_trace_evidence(
    *,
    out_dir: Path,
    scenario: str = "load_step",
    steps: int = 512,
    seed: int = 23,
    controllers: list[str] | None = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces"
    figure_dir = out_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    base_motor, inverter = _make_base_params()
    rng = Random(seed)
    real_params = randomized_motor_params(base_motor, rng)
    lookup = _controller_lookup(quick=False)
    labels = controllers if controllers is not None else list(DEFAULT_TRACE_CONTROLLERS)
    unknown = [label for label in labels if label not in lookup]
    if unknown:
        raise ValueError(f"unknown trace controllers: {unknown}")

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    trace_files: list[str] = []
    for label in labels:
        horizon, feedback_period = lookup[label]
        rows, metrics = _simulate_trace(
            label=label,
            horizon=horizon,
            feedback_period=feedback_period,
            scenario=scenario,
            steps=steps,
            base_motor=base_motor,
            real_params=real_params,
            inverter=inverter,
        )
        trace_file = trace_dir / f"{scenario}_{label}.csv"
        _write_csv(trace_file, rows)
        trace_files.append(trace_file.relative_to(out_dir).as_posix())
        all_rows.extend(rows)
        summary_rows.append({"scenario": scenario, "controller": label, **metrics})

    summary_csv = out_dir / "trace_summary.csv"
    _write_csv(summary_csv, summary_rows)
    speed_svg = figure_dir / "fig_trace_speed.svg"
    fft_svg = figure_dir / "fig_trace_fft_thd.svg"
    _speed_trace_svg(all_rows, scenario, speed_svg)
    _spectral_bar_svg(summary_rows, fft_svg)

    ready = bool(summary_rows) and all(float(row["safety_violations"]) == 0.0 for row in summary_rows)
    payload = {
        "status": "host_trace_fft_thd_evidence",
        "hardware_claim": False,
        "trace_evidence_ready": bool(ready),
        "scenario": scenario,
        "steps": int(steps),
        "sample_period_s": float(inverter.t_pwm_s),
        "seed": int(seed),
        "controllers": list(labels),
        "summary": summary_rows,
        "files": [
            "trace_summary.json",
            summary_csv.relative_to(out_dir).as_posix(),
            speed_svg.relative_to(out_dir).as_posix(),
            fft_svg.relative_to(out_dir).as_posix(),
            *trace_files,
        ],
        "interpretation_limits": [
            "host simulation only",
            "THD-like metrics use the dominant FFT bin as the fundamental and are not hardware power-analyzer THD",
            "trace evidence does not prove MCU/HIL/bench readiness",
        ],
    }
    summary_json = out_dir / "trace_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build host trace/FFT/THD-like evidence for Safe Neural Horizon PWM.")
    parser.add_argument("--out-dir", default=".tmp_pytest/safe_neural_horizon_pwm_trace_evidence")
    parser.add_argument("--scenario", default="load_step")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--controllers", default="", help="Comma-separated controller labels. Defaults to the trace evidence set.")
    args = parser.parse_args()

    controllers = [item.strip() for item in str(args.controllers).split(",") if item.strip()] or None
    payload = build_trace_evidence(
        out_dir=Path(args.out_dir).expanduser().resolve(),
        scenario=str(args.scenario),
        steps=int(args.steps),
        seed=int(args.seed),
        controllers=controllers,
    )
    print(f"saved: {Path(args.out_dir).expanduser().resolve()}")
    print(f"trace_evidence_ready: {payload['trace_evidence_ready']}")


if __name__ == "__main__":
    main()
