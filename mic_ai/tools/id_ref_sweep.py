from __future__ import annotations

"""
Sweep id_ref for FOC and report energy vs speed error.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


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
    env = InductionMotorEnv(replace(env_cfg, sim=sim_cfg, foc=foc_cfg))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep id_ref and report energy vs error.")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--scenario", default="speed_step")
    parser.add_argument("--id-ref-min", type=float, default=0.2)
    parser.add_argument("--id-ref-max", type=float, default=2.0)
    parser.add_argument("--id-ref-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--out-dir", default="outputs/id_ref_sweep")
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    args = parser.parse_args()

    env_cfg = make_env_from_config(args.env_config).env_config
    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)
    sim_cfg = replace(env_cfg.sim, scenario_name=str(args.scenario), dt=dt, t_end=t_end)
    env_cfg = replace(env_cfg, sim=sim_cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_refs = np.linspace(float(args.id_ref_min), float(args.id_ref_max), int(max(args.id_ref_steps, 2)))
    rows: List[Dict[str, float]] = []
    best = None

    for id_ref in id_refs:
        series = _simulate_foc(env_cfg, dt, t_end, float(id_ref), bool(args.use_total_power))
        summary = _summarize(series, float(args.window_frac))
        err_limit = max(summary["mean_abs_speed_err"] * (1.0 + float(args.error_tol_rel)), float(args.error_tol_abs))
        row = {
            "id_ref": float(id_ref),
            "mean_err": float(summary["mean_abs_speed_err"]),
            "mean_p_el_pos": float(summary["mean_p_el_pos"]),
            "eta": float(summary["eta"]),
            "err_limit": float(err_limit),
            "err_ok": bool(summary["mean_abs_speed_err"] <= err_limit),
        }
        rows.append(row)
        if row["err_ok"]:
            if best is None or row["mean_p_el_pos"] < best["mean_p_el_pos"]:
                best = row

    report = {
        "env_config": str(args.env_config),
        "scenario": str(args.scenario),
        "use_total_power": bool(args.use_total_power),
        "rows": rows,
        "best": best,
    }
    (out_dir / "id_ref_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
