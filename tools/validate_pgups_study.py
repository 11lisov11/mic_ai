from __future__ import annotations

"""
Validate that the paper's multi-motor study metrics are self-consistent and scientifically sane.

What it checks:
- Recomputes Pвх+ savings from the raw per-step traces stored in the evaluation run directories.
- Confirms that the saved CSV metrics match recomputed values (within a tight tolerance).
- Adds a sanity check: MIC should not "save energy" by doing materially less mechanical work.
  For that we compute load work E_load = ∫ M_load(t) * ω(t) dt using the *scenario definition*
  (the algorithm does NOT see motor parameters; this is only for offline validation).

This script is intended for maintainers and reviewers. It does not require RL checkpoints.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Allow running as `python tools/validate_pgups_study.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mic_ai.core.env import make_env_from_config
from simulation.scenarios import get_scenario


def _default_study_dir() -> Path:
    """
    Prefer the paper's committed data tables when present, otherwise fall back to local outputs.
    """

    paper = Path("paper/pgups_2026/data")
    if (paper / "scenario_metrics_multi_motor.csv").exists() and (paper / "study_summary_multi_motor.json").exists():
        return paper
    return Path("outputs/research20260214/multi_motor_study")


STUDY_DIR = _default_study_dir()
SUMMARY_JSON = STUDY_DIR / "study_summary_multi_motor.json"
SCENARIO_CSV = STUDY_DIR / "scenario_metrics_multi_motor.csv"

WINDOW_FRAC = 0.30


@dataclass(frozen=True)
class MotorInfo:
    motor_key: str
    motor_label: str
    config: str
    run_dir: Path


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size <= 1:
        return float(np.sum(y))
    return float(np.trapezoid(y, x))


def _steady_slice(n: int, frac: float) -> slice:
    frac = float(max(min(frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - frac)))
    return slice(start, n)


def _steady_slice_by_scenario(t: np.ndarray, omega_ref: np.ndarray, scenario: str, frac: float) -> slice:
    """
    Same definition as tools/multi_motor_study_report.py.

    Special case: `start_stop` ends with deceleration to zero; the final window is not "steady".
    We detect the near-constant plateau around max |ω_ref| and take the last `frac` of that plateau.
    """

    n = int(t.size)
    if n <= 0:
        return slice(0, 0)

    base = _steady_slice(n, frac)
    if str(scenario).strip().lower() != "start_stop" or n < 10:
        return base

    omega_ref_abs = np.abs(omega_ref)
    omega_ref_max = float(np.max(omega_ref_abs))
    if not np.isfinite(omega_ref_max) or omega_ref_max <= 1e-9:
        return base

    mask = omega_ref_abs >= 0.95 * omega_ref_max
    if n >= 2:
        domega_ref_dt = np.abs(np.gradient(omega_ref, t))
        mask &= domega_ref_dt <= 1.0
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return base

    start = int(idx[0])
    end = int(idx[-1]) + 1
    m = int(end - start)
    if m <= 0:
        return base
    w = _steady_slice(m, frac)
    return slice(start + w.start, start + w.stop)


def _pct_saving(base: float, alt: float) -> float:
    return 100.0 * (1.0 - float(alt) / max(float(base), 1e-12))


def _read_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _recompute_savings(df_foc: pd.DataFrame, df_mic: pd.DataFrame, scenario: str) -> Dict[str, float]:
    # Required columns are written by scenario_compare._save_csv().
    t = df_foc["t"].to_numpy(dtype=float)
    omega = df_foc["omega"].to_numpy(dtype=float)
    omega_ref = df_foc["omega_ref"].to_numpy(dtype=float)
    p_foc_signed = df_foc["p_el"].to_numpy(dtype=float)
    p_mic_signed = df_mic["p_el"].to_numpy(dtype=float)
    p_foc = np.maximum(p_foc_signed, 0.0)
    p_mic = np.maximum(p_mic_signed, 0.0)

    n = int(t.size)
    sl = _steady_slice_by_scenario(t, omega_ref, scenario, WINDOW_FRAC)

    out: Dict[str, float] = {}
    # Signed (non-clipped) power is useful to verify that savings are not an artifact of clipping P<0.
    out["foc_mean_p_el_signed_full"] = float(np.mean(p_foc_signed)) if n else 0.0
    out["mic_mean_p_el_signed_full"] = float(np.mean(p_mic_signed)) if n else 0.0
    out["saving_full_pct_signed"] = _pct_saving(out["foc_mean_p_el_signed_full"], out["mic_mean_p_el_signed_full"])

    out["foc_mean_p_el_signed_steady"] = float(np.mean(p_foc_signed[sl])) if n else 0.0
    out["mic_mean_p_el_signed_steady"] = float(np.mean(p_mic_signed[sl])) if n else 0.0
    out["saving_steady_pct_signed"] = _pct_saving(out["foc_mean_p_el_signed_steady"], out["mic_mean_p_el_signed_steady"])

    out["foc_neg_frac_p_el_full"] = float(np.mean(p_foc_signed < 0.0)) if n else 0.0
    out["mic_neg_frac_p_el_full"] = float(np.mean(p_mic_signed < 0.0)) if n else 0.0
    out["foc_neg_frac_p_el_steady"] = float(np.mean(p_foc_signed[sl] < 0.0)) if n else 0.0
    out["mic_neg_frac_p_el_steady"] = float(np.mean(p_mic_signed[sl] < 0.0)) if n else 0.0

    out["foc_mean_p_in_full"] = float(np.mean(p_foc)) if n else 0.0
    out["mic_mean_p_in_full"] = float(np.mean(p_mic)) if n else 0.0
    out["saving_full_pct"] = _pct_saving(out["foc_mean_p_in_full"], out["mic_mean_p_in_full"])

    out["foc_mean_p_in_steady"] = float(np.mean(p_foc[sl])) if n else 0.0
    out["mic_mean_p_in_steady"] = float(np.mean(p_mic[sl])) if n else 0.0
    out["saving_steady_pct"] = _pct_saving(out["foc_mean_p_in_steady"], out["mic_mean_p_in_steady"])

    # Tracking error (MAE).
    err_f = np.abs(df_foc["omega_ref"].to_numpy(dtype=float) - df_foc["omega"].to_numpy(dtype=float))
    err_m = np.abs(df_mic["omega_ref"].to_numpy(dtype=float) - df_mic["omega"].to_numpy(dtype=float))
    out["mae_full_foc"] = float(np.mean(err_f)) if n else 0.0
    out["mae_full_mic"] = float(np.mean(err_m)) if n else 0.0
    out["mae_ratio_full"] = float(out["mae_full_mic"] / max(out["mae_full_foc"], 1e-12))

    out["mae_steady_foc"] = float(np.mean(err_f[sl])) if n else 0.0
    out["mae_steady_mic"] = float(np.mean(err_m[sl])) if n else 0.0
    out["mae_ratio_steady"] = float(out["mae_steady_mic"] / max(out["mae_steady_foc"], 1e-12))

    # Mechanical work sanity: E_load = ∫ M_load(t)*ω(t) dt from the scenario definition.
    # (Use ω from each trace; load depends only on time in our protocol.)
    out["e_load_full_foc_j"] = 0.0
    out["e_load_full_mic_j"] = 0.0
    out["e_load_steady_foc_j"] = 0.0
    out["e_load_steady_mic_j"] = 0.0

    # We reconstruct M_load(t) using the config's scenario generator.
    # If scenario is unknown, we skip this check.
    try:
        # NOTE: omega/load are in SI units in the traces.
        # Any sign changes are handled by clipping negative work to zero (work delivered to the load).
        # This is a conservative check: MIC must not reduce delivered work.
        motor_env_cfg = df_foc.attrs.get("env_cfg")  # not set, kept for future
    except Exception:
        motor_env_cfg = None
    _ = motor_env_cfg  # silence linter; actual env is provided by caller via closure.

    return out


def _load_motor_infos(summary_json: Path) -> Dict[str, MotorInfo]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    motors = payload.get("motors", [])
    out: Dict[str, MotorInfo] = {}
    for m in motors:
        key = str(m["motor_key"])
        out[key] = MotorInfo(
            motor_key=key,
            motor_label=str(m.get("motor_label", key)),
            config=str(m["config"]),
            run_dir=Path(str(m["run_dir"])),
        )
    return out


def _load_env_and_load_func(config_path: str, scenario: str):
    env_cfg = make_env_from_config(config_path).env_config
    omega_ref_func, load_func = get_scenario(str(scenario), env_cfg)
    return env_cfg, omega_ref_func, load_func


def _compute_load_work(t: np.ndarray, omega: np.ndarray, load_func, steady_slice: slice) -> Tuple[float, float]:
    load = np.asarray([float(load_func(float(tt))) for tt in t], dtype=float)
    p_load = load * omega
    e_full = _trapz(np.maximum(p_load, 0.0), t)
    t_w = t[steady_slice]
    e_steady = _trapz(np.maximum(p_load[steady_slice], 0.0), t_w) if t_w.size else 0.0
    return float(e_full), float(e_steady)


def main() -> None:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(SUMMARY_JSON)
    if not SCENARIO_CSV.exists():
        raise FileNotFoundError(SCENARIO_CSV)

    motors = _load_motor_infos(SUMMARY_JSON)
    df = pd.read_csv(SCENARIO_CSV)

    tol_pct = 1e-6  # absolute percentage points; traces are deterministic, so this can be tight

    rows_out = []
    issues = []

    for _, r in df.iterrows():
        motor_key = str(r["motor_key"])
        scenario = str(r["scenario"])
        file_tag = str(r["file_tag"])
        mi = motors.get(motor_key)
        if mi is None:
            issues.append(f"Missing motor mapping for motor_key={motor_key}")
            continue

        foc_path = mi.run_dir / f"{file_tag}_foc.csv"
        mic_path = mi.run_dir / f"{file_tag}_mic_ai.csv"
        foc = _read_trace(foc_path)
        mic = _read_trace(mic_path)

        # Basic alignment checks.
        if foc.shape[0] != mic.shape[0]:
            issues.append(f"{motor_key}/{scenario}: trace length mismatch foc={foc.shape[0]} mic={mic.shape[0]}")
        # omega_ref is generated independently for each run; due to float rounding in CSV it may differ by ~1e-5.
        if not np.allclose(
            foc["omega_ref"].to_numpy(float),
            mic["omega_ref"].to_numpy(float),
            atol=1e-5,
            rtol=0.0,
        ):
            issues.append(f"{motor_key}/{scenario}: omega_ref mismatch between FOC and MIC traces")

        # Recompute savings.
        rec = _recompute_savings(foc, mic, scenario=scenario)

        # Load-work sanity check.
        env_cfg, _omega_ref_func, load_func = _load_env_and_load_func(mi.config, scenario)
        t = foc["t"].to_numpy(dtype=float)
        omega_f = foc["omega"].to_numpy(dtype=float)
        omega_m = mic["omega"].to_numpy(dtype=float)
        sl = _steady_slice_by_scenario(t, foc["omega_ref"].to_numpy(dtype=float), scenario, WINDOW_FRAC)
        e_load_f_full, e_load_f_steady = _compute_load_work(t, omega_f, load_func, sl)
        e_load_m_full, e_load_m_steady = _compute_load_work(t, omega_m, load_func, sl)

        rec["e_load_full_foc_j"] = e_load_f_full
        rec["e_load_full_mic_j"] = e_load_m_full
        rec["e_load_steady_foc_j"] = e_load_f_steady
        rec["e_load_steady_mic_j"] = e_load_m_steady
        rec["load_work_ratio_full"] = float(e_load_m_full / max(e_load_f_full, 1e-12))
        rec["load_work_ratio_steady"] = float(e_load_m_steady / max(e_load_f_steady, 1e-12))

        # Compare with CSV.
        for key_csv, key_rec in (("saving_full_pct", "saving_full_pct"), ("saving_steady_pct", "saving_steady_pct"), ("mae_ratio_full", "mae_ratio_full")):
            v_csv = float(r[key_csv])
            v_rec = float(rec[key_rec])
            if not np.isfinite(v_csv) or not np.isfinite(v_rec) or abs(v_csv - v_rec) > tol_pct:
                issues.append(
                    f"{motor_key}/{scenario}: mismatch {key_csv}: csv={v_csv:.9f} rec={v_rec:.9f} (tol={tol_pct})"
                )

        # Scientific sanity flags.
        if rec["load_work_ratio_steady"] < 0.99:
            issues.append(
                f"{motor_key}/{scenario}: MIC delivers less load work in steady window: "
                f"ratio={rec['load_work_ratio_steady']:.4f}"
            )
        if rec["mae_ratio_full"] > 1.05:
            issues.append(f"{motor_key}/{scenario}: speed MAE worsens vs FOC: ratio={rec['mae_ratio_full']:.4f}")

        rows_out.append(
            {
                "motor_key": motor_key,
                "scenario": scenario,
                "saving_full_pct_csv": float(r["saving_full_pct"]),
                "saving_full_pct_rec": float(rec["saving_full_pct"]),
                "saving_steady_pct_csv": float(r["saving_steady_pct"]),
                "saving_steady_pct_rec": float(rec["saving_steady_pct"]),
                "saving_full_pct_signed": float(rec["saving_full_pct_signed"]),
                "saving_steady_pct_signed": float(rec["saving_steady_pct_signed"]),
                "foc_neg_frac_p_el_full": float(rec["foc_neg_frac_p_el_full"]),
                "mic_neg_frac_p_el_full": float(rec["mic_neg_frac_p_el_full"]),
                "foc_neg_frac_p_el_steady": float(rec["foc_neg_frac_p_el_steady"]),
                "mic_neg_frac_p_el_steady": float(rec["mic_neg_frac_p_el_steady"]),
                "mae_ratio_full_csv": float(r["mae_ratio_full"]),
                "mae_ratio_full_rec": float(rec["mae_ratio_full"]),
                "load_work_ratio_full": float(rec["load_work_ratio_full"]),
                "load_work_ratio_steady": float(rec["load_work_ratio_steady"]),
                "e_load_full_foc_j": float(rec["e_load_full_foc_j"]),
                "e_load_full_mic_j": float(rec["e_load_full_mic_j"]),
                "e_load_steady_foc_j": float(rec["e_load_steady_foc_j"]),
                "e_load_steady_mic_j": float(rec["e_load_steady_mic_j"]),
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_path = STUDY_DIR / "validation_report_pgups_2026.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"OK: wrote {out_path}")
    if issues:
        print("\nISSUES:")
        for msg in issues:
            print(f"- {msg}")
        raise SystemExit(2)
    print("OK: all checks passed")


if __name__ == "__main__":
    main()
