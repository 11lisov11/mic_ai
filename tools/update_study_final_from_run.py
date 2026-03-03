from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SCENARIOS = ("hold:0.8", "speed_step", "ramp", "load_step", "start_stop")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _steady_slice(n: int, frac: float) -> slice:
    frac = float(max(min(frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - frac)))
    return slice(start, n)


def _series_metrics(series: pd.DataFrame, window_frac: float) -> Dict[str, float]:
    t = series["t"].to_numpy(dtype=float)
    omega = series["omega"].to_numpy(dtype=float)
    omega_ref = series["omega_ref"].to_numpy(dtype=float)
    p_in_pos = np.maximum(series["p_el"].to_numpy(dtype=float), 0.0)
    p_shaft_pos = np.maximum(series["p_mech"].to_numpy(dtype=float), 0.0)
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
        "eta_full": eta_full,
        "mae_speed_full": float(np.mean(err)) if err.size else 0.0,
        "mean_p_in_steady": float(np.mean(p_in_w)) if p_in_w.size else 0.0,
        "mean_p_shaft_steady": float(np.mean(p_shaft_w)) if p_shaft_w.size else 0.0,
        "eta_steady": eta_w,
        "mae_speed_steady": float(np.mean(err_w)) if err_w.size else 0.0,
    }


def _scenario_table(final_best: Path, window_frac: float) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for scenario in SCENARIOS:
        tag = scenario.replace(":", "_").replace(".", "p")
        if scenario == "load_step":
            # Backward compatibility with old trace naming.
            fallback = final_best / "load_profile_foc.csv"
            if fallback.exists():
                tag = "load_profile"
        foc = _read_csv(final_best / f"{tag}_foc.csv")
        mic = _read_csv(final_best / f"{tag}_mic_ai.csv")
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


def _write_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")


def _copy_run_to_final_best(run_dir: Path, final_best: Path) -> None:
    final_best.mkdir(parents=True, exist_ok=True)
    # Keep backup for reproducibility.
    backup = final_best.parent / "final_best_prev"
    if final_best.exists() and any(final_best.iterdir()):
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(final_best, backup)

    for p in run_dir.glob("*"):
        if p.is_file():
            shutil.copy2(p, final_best / p.name)


def _load_compare_summary(path: Path) -> List[Dict[str, float | str | bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list summary in {path}")
    return data


def main() -> None:
    warnings.warn(
        "tools/update_study_final_from_run.py is a legacy helper. "
        "For reproducible IEEE artifacts use tools/reproduce_ieee_step28.py.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Update study_final artifacts from a scenario_compare run directory.")
    parser.add_argument("--run-dir", required=True, help="Directory containing *_foc.csv, *_mic_ai.csv and summary.json")
    parser.add_argument("--study-dir", default="paper/pgups_2026/data")
    parser.add_argument("--window-frac", type=float, default=0.3)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path used for this run (for study_summary.json)")
    parser.add_argument("--method", default="MIC AI eta-supervisor")
    parser.add_argument("--max-mae-pass", type=float, default=1.06)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    study_dir = Path(args.study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    final_best = study_dir / "final_best"
    _copy_run_to_final_best(run_dir, final_best)

    scenario_rows = _scenario_table(final_best, window_frac=float(args.window_frac))
    _write_csv(scenario_rows, study_dir / "scenario_metrics.csv")

    # Energy balance table (full cycle).
    eb_rows = []
    for r in scenario_rows:
        eb_rows.append(
            {
                "scenario": str(r["file_tag"]),
                "foc_pin_w": float(r["foc_p_in_full"]),
                "mic_pin_w": float(r["mic_p_in_full"]),
                "saving_pct": float(r["p_in_saving_full_pct"]),
                "foc_p2_w": float(r["foc_p_shaft_full"]),
                "mic_p2_w": float(r["mic_p_shaft_full"]),
                "foc_eta": float(r["foc_eta_full"]),
                "mic_eta": float(r["mic_eta_full"]),
                "eta_gain_pct": float(r["eta_gain_full_pct"]),
            }
        )
    _write_csv(eb_rows, study_dir / "energy_balance_emax120_fullcycle.csv")

    compare_summary = _load_compare_summary(final_best / "summary.json")
    mae_ratios = [float(r["mic_mean_err"]) / max(float(r["foc_mean_err"]), 1e-9) for r in compare_summary]
    all_err_ok = all(bool(r.get("err_ok", False)) for r in compare_summary)

    avg_save_full = float(np.mean([float(r["p_in_saving_full_pct"]) for r in scenario_rows]))
    min_save_full = float(np.min([float(r["p_in_saving_full_pct"]) for r in scenario_rows]))
    avg_eta_full = float(np.mean([float(r["eta_gain_full_pct"]) for r in scenario_rows]))
    min_eta_full = float(np.min([float(r["eta_gain_full_pct"]) for r in scenario_rows]))
    avg_save_steady = float(np.mean([float(r["p_in_saving_steady_pct"]) for r in scenario_rows]))
    avg_eta_steady = float(
        np.mean(
            [
                100.0 * (float(r["mic_eta_steady"]) / max(float(r["foc_eta_steady"]), 1e-9) - 1.0)
                for r in scenario_rows
            ]
        )
    )
    max_mae = float(np.max(mae_ratios))
    avg_mae = float(np.mean(mae_ratios))

    study_summary = {
        "method": str(args.method),
        "checkpoint": args.checkpoint,
        "run_dir": str(run_dir),
        "avg_power_saving_full_pct": avg_save_full,
        "avg_power_saving_steady_pct": avg_save_steady,
        "worst_power_saving_full_pct": min_save_full,
        "avg_eta_gain_full_pct": avg_eta_full,
        "avg_eta_gain_steady_pct": avg_eta_steady,
        "worst_eta_gain_full_pct": min_eta_full,
        "avg_mae_ratio_full": avg_mae,
        "max_mae_ratio_full": max_mae,
        "all_err_ok": bool(all_err_ok),
        "scenarios": list(SCENARIOS),
    }
    (study_dir / "study_summary.json").write_text(json.dumps(study_summary, indent=2), encoding="utf-8")

    significant = {
        "criteria": {
            "min_saving_pct": 1.0,
            "avg_saving_pct": 1.5,
            "avg_eta_gain_pct": 0.0,
            "avg_eta_gain_steady_pct": 0.0,
            "max_mae_ratio": float(args.max_mae_pass),
            "all_err_ok": True,
        },
        "values": {
            "avg_saving_pct": avg_save_full,
            "min_saving_pct": min_save_full,
            "avg_eta_gain_pct": avg_eta_full,
            "avg_eta_gain_steady_pct": avg_eta_steady,
            "max_mae_ratio": max_mae,
            "all_err_ok": bool(all_err_ok),
        },
        "passed": bool(
            avg_save_full >= 1.5
            and min_save_full >= 1.0
            and max_mae <= float(args.max_mae_pass)
            and all_err_ok
        ),
        "passed_with_eta_steady": bool(
            avg_save_full >= 1.5
            and min_save_full >= 1.0
            and avg_eta_steady >= 0.0
            and max_mae <= float(args.max_mae_pass)
            and all_err_ok
        ),
        "passed_with_eta_full": bool(
            avg_save_full >= 1.5
            and min_save_full >= 1.0
            and avg_eta_full >= 0.0
            and max_mae <= float(args.max_mae_pass)
            and all_err_ok
        ),
    }
    (study_dir / "significant_gain_check.json").write_text(json.dumps(significant, indent=2), encoding="utf-8")

    print(f"Updated study directory: {study_dir}")
    print(json.dumps(study_summary, indent=2))


if __name__ == "__main__":
    main()
