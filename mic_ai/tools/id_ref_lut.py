from __future__ import annotations

"""
Build and evaluate id_ref LUT as a function of omega_ref and load_torque.
"""

import argparse
import json
import copy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


def _clone_env_cfg(env_cfg: object) -> object:
    try:
        return copy.copy(env_cfg)
    except Exception:
        return env_cfg


def _set_attr(obj: object, name: str, value: object) -> bool:
    try:
        object.__setattr__(obj, name, value)
        return True
    except Exception:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False


def _clone_with(env_cfg: object, **kwargs: object) -> object:
    env_cfg_s = _clone_env_cfg(env_cfg)
    for key, val in kwargs.items():
        if not _set_attr(env_cfg_s, key, val):
            env_cfg_s = replace(env_cfg_s, **{key: val})
    return env_cfg_s


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _summarize(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    p_el = series["p_el"][sl]
    p_mech = series["p_mech"][sl]
    err = np.abs(series["omega_ref"][sl] - series["omega"][sl])
    p_el_mean = float(np.mean(p_el)) if p_el.size else 0.0
    p_el_pos_mean = float(np.mean(np.maximum(p_el, 0.0))) if p_el.size else 0.0
    p_mech_mean = float(np.mean(p_mech)) if p_mech.size else 0.0
    eta = float(p_mech_mean / p_el_mean) if p_el_mean > 1e-9 else 0.0
    return {
        "omega_ss": float(np.mean(series["omega"][sl])) if n else 0.0,
        "mean_abs_speed_err": float(np.mean(err)) if err.size else 0.0,
        "mean_p_el": p_el_mean,
        "mean_p_el_pos": p_el_pos_mean,
        "p_mech": p_mech_mean,
        "eta": eta,
    }


def _simulate_foc(env_cfg: object, dt: float, t_end: float, id_ref: float, use_total_power: bool) -> Dict[str, np.ndarray]:
    sim_cfg = replace(env_cfg.sim, dt=dt, t_end=t_end)
    foc_cfg = replace(env_cfg.foc, id_ref=float(id_ref))
    env_cfg_s = _clone_with(env_cfg, sim=sim_cfg, foc=foc_cfg)
    env = InductionMotorEnv(env_cfg_s)
    env.reset()
    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0])
        omega_ref[k] = float(obs[1])
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2]))
        i_rms[k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        if use_total_power:
            p_el_val = float(info.get("p_in_total", p_el_val))
        p_el[k] = p_el_val
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "omega_ref": omega_ref,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _parse_range(text: str) -> Tuple[float, float]:
    raw = str(text).strip().replace(":", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Range must be 'min,max'")
    lo = float(parts[0])
    hi = float(parts[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _grid(lo: float, hi: float, n: int) -> List[float]:
    return [float(x) for x in np.linspace(lo, hi, int(max(n, 2)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build id_ref LUT over omega_ref and load torque.")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--omega-ref-range", required=True, help="min,max rad/s or pu when --omega-ref-pu")
    parser.add_argument("--omega-ref-pu", action="store_true", help="Interpret omega_ref_range in per-unit.")
    parser.add_argument("--omega-ref-steps", type=int, default=5)
    parser.add_argument("--load-range", required=True, help="min,max load torque (N*m).")
    parser.add_argument("--load-steps", type=int, default=5)
    parser.add_argument("--id-ref-min", type=float, default=0.2)
    parser.add_argument("--id-ref-max", type=float, default=2.0)
    parser.add_argument("--id-ref-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--out-dir", default="outputs/id_ref_lut")
    args = parser.parse_args()

    env_cfg = make_env_from_config(args.env_config).env_config
    if getattr(env_cfg, "id_ref_lut_path", None) is not None:
        _set_attr(env_cfg, "id_ref_lut_path", None)
    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)
    omega_lo, omega_hi = _parse_range(args.omega_ref_range)
    if args.omega_ref_pu:
        omega_base = float(2.0 * np.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p)
        omega_lo *= omega_base
        omega_hi *= omega_base
    load_lo, load_hi = _parse_range(args.load_range)

    omega_grid = _grid(omega_lo, omega_hi, int(args.omega_ref_steps))
    load_grid = _grid(load_lo, load_hi, int(args.load_steps))
    id_grid = _grid(float(args.id_ref_min), float(args.id_ref_max), int(args.id_ref_steps))

    rows: List[Dict[str, float]] = []
    best_map: Dict[str, float] = {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omega_nom = float(2.0 * np.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p)
    base_id_ref = float(env_cfg.foc.id_ref)
    if not any(abs(float(x) - base_id_ref) < 1e-6 for x in id_grid):
        id_grid.append(base_id_ref)
        id_grid = sorted(set(float(x) for x in id_grid))

    for omega_ref in omega_grid:
        omega_pu = float(omega_ref / omega_nom) if omega_nom > 1e-6 else 0.0
        scenario_name = f"hold:{omega_pu:.6g}"
        for load in load_grid:
            sim_cfg = replace(env_cfg.sim, scenario_name=scenario_name, dt=dt, t_end=t_end, load_torque=float(load))
            env_cfg_case = _clone_with(env_cfg, sim=sim_cfg)
            base_series = _simulate_foc(env_cfg_case, dt, t_end, base_id_ref, bool(args.use_total_power))
            base_summary = _summarize(base_series, float(args.window_frac))
            err_limit = max(
                float(base_summary["mean_abs_speed_err"]) * (1.0 + float(args.error_tol_rel)),
                float(args.error_tol_abs),
            )
            best = None
            for id_ref in id_grid:
                series = _simulate_foc(env_cfg_case, dt, t_end, float(id_ref), bool(args.use_total_power))
                summary = _summarize(series, float(args.window_frac))
                entry = {
                    "omega_ref": float(omega_ref),
                    "load_torque": float(load),
                    "id_ref": float(id_ref),
                    "mean_err": float(summary["mean_abs_speed_err"]),
                    "mean_p_el_pos": float(summary["mean_p_el_pos"]),
                    "eta": float(summary["eta"]),
                    "err_limit": float(err_limit),
                    "err_ok": bool(summary["mean_abs_speed_err"] <= err_limit),
                }
                rows.append(entry)
                if entry["err_ok"]:
                    if best is None or entry["mean_p_el_pos"] < best["mean_p_el_pos"]:
                        best = entry
            if best is None:
                best = {
                    "omega_ref": float(omega_ref),
                    "load_torque": float(load),
                    "id_ref": float(id_grid[0]),
                    "mean_err": 0.0,
                    "mean_p_el_pos": 0.0,
                    "eta": 0.0,
                    "err_limit": 0.0,
                    "err_ok": False,
                }
            key = f"{omega_ref:.6g}|{load:.6g}"
            best_map[key] = float(best["id_ref"])

    report = {
        "env_config": str(args.env_config),
        "omega_ref_grid": omega_grid,
        "load_grid": load_grid,
        "id_ref_grid": id_grid,
        "use_total_power": bool(args.use_total_power),
        "rows": rows,
        "lut": best_map,
    }
    (out_dir / "id_ref_lut.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
