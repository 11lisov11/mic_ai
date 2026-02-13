from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure


@dataclass
class CandidateResult:
    id_ref: float
    mean_power_saving_pct: float
    min_power_saving_pct: float
    max_err_ratio: float
    summary_path: Path
    out_dir: Path


def _run_compare(
    python_exe: str,
    env_config: str,
    scenarios: str,
    t_end: float,
    dt: float,
    window_frac: float,
    out_dir: Path,
    id_ref: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "-m",
        "mic_ai.tools.scenario_compare",
        "--env-config",
        env_config,
        "--scenarios",
        scenarios,
        "--t-end",
        str(t_end),
        "--dt",
        str(dt),
        "--window-frac",
        str(window_frac),
        "--error-tol-rel",
        "0.2",
        "--error-tol-abs",
        "0.0",
        "--use-total-power",
        "--foc-disable-lut",
        "--mic-id-ref",
        str(id_ref),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    summary = out_dir / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Missing summary: {summary}")
    return summary


def _load_summary(path: Path) -> List[Dict[str, float | str | bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"summary must be list: {path}")
    return data


def _evaluate_candidate(id_ref: float, summary_path: Path, out_dir: Path) -> CandidateResult:
    rows = _load_summary(summary_path)
    savings: List[float] = []
    err_ratios: List[float] = []
    for row in rows:
        foc_err = float(row["foc_mean_err"])
        mic_err = float(row["mic_mean_err"])
        err_ratios.append(mic_err / max(foc_err, 1e-9))
        savings.append(float(row["power_saving_pct"]))
    return CandidateResult(
        id_ref=float(id_ref),
        mean_power_saving_pct=float(np.mean(savings)) if savings else 0.0,
        min_power_saving_pct=float(np.min(savings)) if savings else 0.0,
        max_err_ratio=float(np.max(err_ratios)) if err_ratios else 0.0,
        summary_path=summary_path,
        out_dir=out_dir,
    )


def _pick_best(
    candidates: Sequence[CandidateResult],
    max_err_ratio: float,
    require_non_negative_worst: bool,
) -> CandidateResult:
    filtered = [
        c
        for c in candidates
        if c.max_err_ratio <= max_err_ratio and (not require_non_negative_worst or c.min_power_saving_pct >= 0.0)
    ]
    pool = filtered if filtered else list(candidates)
    pool.sort(key=lambda c: (c.mean_power_saving_pct, c.min_power_saving_pct), reverse=True)
    return pool[0]


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols: Dict[str, List[float]] = {}
        for row in reader:
            for key, value in row.items():
                cols.setdefault(key, []).append(float(value))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _steady_slice(n: int, frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    frac = float(max(min(frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - frac)))
    return slice(start, n)


def _series_metrics(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    t = series["t"]
    omega = series["omega"]
    omega_ref = series["omega_ref"]
    p_in_pos = np.maximum(series["p_el"], 0.0)
    p_shaft_pos = np.maximum(series["p_mech"], 0.0)
    err = np.abs(omega_ref - omega)

    n = int(t.size)
    sl = _steady_slice(n, window_frac)

    def _trapz(y: np.ndarray, x: np.ndarray) -> float:
        return float(np.trapezoid(y, x)) if y.size and x.size else 0.0

    e_in_full = _trapz(p_in_pos, t)
    e_shaft_full = _trapz(p_shaft_pos, t)
    eta_full = e_shaft_full / max(e_in_full, 1e-9)

    t_w = t[sl]
    p_in_w = p_in_pos[sl]
    p_shaft_w = p_shaft_pos[sl]
    err_w = err[sl]
    e_in_w = _trapz(p_in_w, t_w) if t_w.size > 1 else float(np.sum(p_in_w))
    e_shaft_w = _trapz(p_shaft_w, t_w) if t_w.size > 1 else float(np.sum(p_shaft_w))
    eta_w = e_shaft_w / max(e_in_w, 1e-9)

    return {
        "mean_p_in_full": float(np.mean(p_in_pos)) if p_in_pos.size else 0.0,
        "mean_p_shaft_full": float(np.mean(p_shaft_pos)) if p_shaft_pos.size else 0.0,
        "energy_in_full": e_in_full,
        "energy_shaft_full": e_shaft_full,
        "eta_full": eta_full,
        "mae_speed_full": float(np.mean(err)) if err.size else 0.0,
        "mean_p_in_steady": float(np.mean(p_in_w)) if p_in_w.size else 0.0,
        "mean_p_shaft_steady": float(np.mean(p_shaft_w)) if p_shaft_w.size else 0.0,
        "energy_in_steady": e_in_w,
        "energy_shaft_steady": e_shaft_w,
        "eta_steady": eta_w,
        "mae_speed_steady": float(np.mean(err_w)) if err_w.size else 0.0,
    }


def _scenario_table(final_out_dir: Path, window_frac: float) -> List[Dict[str, float | str]]:
    summary = _load_summary(final_out_dir / "summary.json")
    rows: List[Dict[str, float | str]] = []
    for item in summary:
        tag = str(item["file_tag"])
        scenario = str(item["scenario"])
        foc = _read_csv(final_out_dir / f"{tag}_foc.csv")
        mic = _read_csv(final_out_dir / f"{tag}_mic_ai.csv")
        m_foc = _series_metrics(foc, window_frac)
        m_mic = _series_metrics(mic, window_frac)

        def _pct_gain(a: float, b: float) -> float:
            return 100.0 * (1.0 - b / max(a, 1e-9))

        rows.append(
            {
                "scenario": scenario,
                "foc_p_in_full": m_foc["mean_p_in_full"],
                "mic_p_in_full": m_mic["mean_p_in_full"],
                "p_in_saving_full_pct": _pct_gain(m_foc["mean_p_in_full"], m_mic["mean_p_in_full"]),
                "foc_p_shaft_full": m_foc["mean_p_shaft_full"],
                "mic_p_shaft_full": m_mic["mean_p_shaft_full"],
                "foc_eta_full": m_foc["eta_full"],
                "mic_eta_full": m_mic["eta_full"],
                "eta_gain_full_pct": 100.0 * (m_mic["eta_full"] / max(m_foc["eta_full"], 1e-9) - 1.0),
                "foc_mae_full": m_foc["mae_speed_full"],
                "mic_mae_full": m_mic["mae_speed_full"],
                "foc_p_in_steady": m_foc["mean_p_in_steady"],
                "mic_p_in_steady": m_mic["mean_p_in_steady"],
                "p_in_saving_steady_pct": _pct_gain(m_foc["mean_p_in_steady"], m_mic["mean_p_in_steady"]),
                "foc_eta_steady": m_foc["eta_steady"],
                "mic_eta_steady": m_mic["eta_steady"],
                "foc_mae_steady": m_foc["mae_speed_steady"],
                "mic_mae_steady": m_mic["mae_speed_steady"],
                "file_tag": tag,
            }
        )
    return rows


def _save_table(rows: Iterable[Dict[str, float | str]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_power_saving(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    scenarios = [str(r["scenario"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    full = np.asarray([float(r["p_in_saving_full_pct"]) for r in rows], dtype=float)
    steady = np.asarray([float(r["p_in_saving_steady_pct"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - width / 2, full, width, label="Full cycle")
    ax.bar(x + width / 2, steady, width, label="Steady window")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Power saving, %")
    ax.set_title("FOC vs MIC: Input Power Saving by Scenario")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_power_eta(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    scenarios = [str(r["scenario"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    foc_pin = np.asarray([float(r["foc_p_in_full"]) for r in rows], dtype=float)
    mic_pin = np.asarray([float(r["mic_p_in_full"]) for r in rows], dtype=float)
    foc_ps = np.asarray([float(r["foc_p_shaft_full"]) for r in rows], dtype=float)
    mic_ps = np.asarray([float(r["mic_p_shaft_full"]) for r in rows], dtype=float)
    foc_eta = np.asarray([float(r["foc_eta_full"]) for r in rows], dtype=float)
    mic_eta = np.asarray([float(r["mic_eta_full"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4), sharex=True)
    axes[0].bar(x - width / 2, foc_pin, width, label="FOC")
    axes[0].bar(x + width / 2, mic_pin, width, label="MIC")
    axes[0].set_ylabel("P_in+, W")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, foc_ps, width, label="FOC")
    axes[1].bar(x + width / 2, mic_ps, width, label="MIC")
    axes[1].set_ylabel("P_shaft+, W")

    axes[2].bar(x - width / 2, foc_eta, width, label="FOC")
    axes[2].bar(x + width / 2, mic_eta, width, label="MIC")
    axes[2].set_ylabel("eta, p.u.")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scenarios, rotation=25, ha="right")
    axes[2].set_xlabel("Scenario")

    fig.suptitle("Input Power, Shaft Power and Efficiency (Full Cycle)")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_timeseries(final_out_dir: Path, file_tag: str, out_path: Path) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    foc = _read_csv(final_out_dir / f"{file_tag}_foc.csv")
    mic = _read_csv(final_out_dir / f"{file_tag}_mic_ai.csv")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(foc["t"], foc["omega"], color="black", label="FOC")
    axes[0].plot(mic["t"], mic["omega"], color="0.35", linestyle="--", label="MIC")
    axes[0].plot(foc["t"], foc["omega_ref"], color="tab:blue", linewidth=1.2, alpha=0.7, label="omega_ref")
    axes[0].set_ylabel("omega, rad/s")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(foc["t"], np.maximum(foc["p_el"], 0.0), color="black", label="FOC")
    axes[1].plot(mic["t"], np.maximum(mic["p_el"], 0.0), color="0.35", linestyle="--", label="MIC")
    axes[1].set_ylabel("P_in+, W")

    axes[2].plot(foc["t"], np.maximum(foc["p_mech"], 0.0), color="black", label="FOC")
    axes[2].plot(mic["t"], np.maximum(mic["p_mech"], 0.0), color="0.35", linestyle="--", label="MIC")
    axes[2].set_ylabel("P_shaft+, W")
    axes[2].set_xlabel("t, s")

    fig.suptitle(f"Time Series: {file_tag}")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_algorithm(out_path: Path) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.axis("off")

    boxes = {
        "sensor": (0.05, 0.58, "Sensors:\nomega, i_abc, load"),
        "policy": (0.28, 0.58, "MIC Self-Learning Policy\n(id_ref scheduler)"),
        "inner": (0.52, 0.58, "FOC Current Loops\n(PI id/iq)"),
        "plant": (0.76, 0.58, "Inverter + IM Plant"),
        "reward": (0.28, 0.20, "Reward:\nmin P_in+, keep speed"),
        "update": (0.52, 0.20, "Online/Offline Update:\nsearch + distillation"),
    }
    for x, y, text in boxes.values():
        rect = plt.Rectangle((x, y), 0.18, 0.22, fill=False, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    arrow(0.23, 0.69, 0.28, 0.69)
    arrow(0.46, 0.69, 0.52, 0.69)
    arrow(0.70, 0.69, 0.76, 0.69)
    arrow(0.85, 0.58, 0.14, 0.58)
    arrow(0.37, 0.58, 0.37, 0.42)
    arrow(0.46, 0.31, 0.52, 0.31)
    arrow(0.61, 0.42, 0.37, 0.42)

    ax.set_title("MIC Algorithm Block Diagram")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research pipeline: FOC vs MIC (fixed id_ref candidates).")
    parser.add_argument("--env-config", default="config/env_research_motor1_loss_nominal.py")
    parser.add_argument("--scenarios", default="hold:0.8,speed_step,ramp,load_profile,start_stop")
    parser.add_argument("--candidate-id-refs", default="1.2,1.25,1.3,1.35,1.4")
    parser.add_argument("--max-err-ratio", type=float, default=1.3)
    parser.add_argument("--require-non-negative-worst-saving", action="store_true")
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--window-frac", type=float, default=0.3)
    parser.add_argument("--out-dir", default="outputs/research20260212/study_final")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = [float(x.strip()) for x in str(args.candidate_id_refs).split(",") if x.strip()]
    candidate_results: List[CandidateResult] = []
    for id_ref in candidates:
        tag = str(id_ref).replace(".", "p")
        run_dir = runs_dir / f"id_{tag}"
        summary_path = _run_compare(
            python_exe=str(args.python),
            env_config=str(args.env_config),
            scenarios=str(args.scenarios),
            t_end=float(args.t_end),
            dt=float(args.dt),
            window_frac=float(args.window_frac),
            out_dir=run_dir,
            id_ref=id_ref,
        )
        candidate_results.append(_evaluate_candidate(id_ref, summary_path, run_dir))

    best = _pick_best(
        candidate_results,
        max_err_ratio=float(args.max_err_ratio),
        require_non_negative_worst=bool(args.require_non_negative_worst_saving),
    )

    final_dir = out_dir / "final_best"
    if final_dir.exists():
        # Keep old files for reproducibility, but overwrite final artifacts below.
        pass
    summary_path = _run_compare(
        python_exe=str(args.python),
        env_config=str(args.env_config),
        scenarios=str(args.scenarios),
        t_end=float(args.t_end),
        dt=float(args.dt),
        window_frac=float(args.window_frac),
        out_dir=final_dir,
        id_ref=float(best.id_ref),
    )

    table = _scenario_table(final_dir, float(args.window_frac))
    _save_table(table, out_dir / "scenario_metrics.csv")

    agg = {
        "best_id_ref": float(best.id_ref),
        "candidate_results": [
            {
                "id_ref": c.id_ref,
                "mean_power_saving_pct": c.mean_power_saving_pct,
                "min_power_saving_pct": c.min_power_saving_pct,
                "max_err_ratio": c.max_err_ratio,
                "summary_path": str(c.summary_path),
            }
            for c in candidate_results
        ],
        "final_summary_path": str(summary_path),
        "avg_power_saving_full_pct": float(np.mean([float(r["p_in_saving_full_pct"]) for r in table])) if table else 0.0,
        "worst_power_saving_full_pct": float(np.min([float(r["p_in_saving_full_pct"]) for r in table])) if table else 0.0,
        "avg_eta_gain_full_pct": float(np.mean([float(r["eta_gain_full_pct"]) for r in table])) if table else 0.0,
        "scenarios": [str(r["scenario"]) for r in table],
    }
    (out_dir / "study_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    _plot_power_saving(table, out_dir / "fig_power_saving")
    _plot_power_eta(table, out_dir / "fig_power_eta")
    by_tag = {str(r["file_tag"]): str(r["scenario"]) for r in table}
    if "hold_0p8" in by_tag:
        _plot_timeseries(final_dir, "hold_0p8", out_dir / "fig_timeseries_hold")
    if "start_stop" in by_tag:
        _plot_timeseries(final_dir, "start_stop", out_dir / "fig_timeseries_start_stop")
    _plot_algorithm(out_dir / "fig_algorithm_block")

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
