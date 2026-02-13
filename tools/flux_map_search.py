from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import EnvConfig, create_default_env
from drivers import SimDriver
from control.vector_foc import FocController
from control.load_map_foc import LoadMapParams


def _load_env_config(path: str | None) -> EnvConfig:
    if path is None:
        return create_default_env()
    from mic_ai.core.env import make_env_from_config

    env = make_env_from_config(path)
    return env.env_config


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _run_foc(env_cfg: EnvConfig, dt: float, t_end: float, seed: int) -> Dict[str, np.ndarray]:
    sim = SimDriver(env_cfg)
    sim.reset(seed=seed)
    sim.set_mode("FOC")
    n_steps = int(max(t_end / dt, 1))
    t = np.zeros(n_steps, dtype=float)
    omega = np.zeros(n_steps, dtype=float)
    omega_ref = np.zeros(n_steps, dtype=float)
    p_el = np.zeros(n_steps, dtype=float)
    for k in range(n_steps):
        sim.step()
        obs = sim.read_obs()
        t[k] = obs["t"]
        omega[k] = obs["omega"]
        omega_ref[k] = obs["omega_ref"]
        v_a, v_b, v_c = obs["v_a"], obs["v_b"], obs["v_c"]
        i_a, i_b, i_c = obs["ia"], obs["ib"], obs["ic"]
        p_el[k] = v_a * i_a + v_b * i_b + v_c * i_c
        if sim.get_last_fault():
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            p_el = p_el[: k + 1]
            break
    return {"t": t, "omega": omega, "omega_ref": omega_ref, "p_el": p_el}


def _summarize(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    err = np.abs(series["omega_ref"][sl] - series["omega"][sl])
    p_el = series["p_el"][sl]
    mean_err = float(np.mean(err)) if err.size else 0.0
    mean_p_pos = float(np.mean(np.maximum(p_el, 0.0))) if p_el.size else 0.0
    return {"mean_abs_speed_error": mean_err, "mean_p_in_pos": mean_p_pos}


def _format_table(rows: List[Dict[str, object]]) -> str:
    headers = ["load_pct", "best_id_ref", "FOC_P_W", "BEST_P_W", "saving_pct", "err_ratio"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search optimal id_ref per load at nominal speed.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--omega-pu", type=float, default=1.0)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--err-tol", type=float, default=0.02)
    parser.add_argument("--out-dir", default="outputs/flux_map")
    args = parser.parse_args()

    env_cfg = _load_env_config(args.env_config)
    base_load = float(env_cfg.sim.load_torque)
    sim_cfg_base = replace(env_cfg.sim, scenario_name=f"hold:{args.omega_pu}", dt=float(args.dt), t_end=float(args.t_end))

    # candidate id_ref ratios
    id_ref_base = float(env_cfg.foc.id_ref)
    id_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    load_pcts = [0, 25, 50, 75, 100]

    rows: List[Dict[str, object]] = []
    map_loads: List[float] = []
    map_ids: List[float] = []

    for pct in load_pcts:
        load = base_load * (pct / 100.0)
        sim_cfg = replace(sim_cfg_base, load_torque=float(load))
        cfg = replace(env_cfg, sim=sim_cfg)

        foc_series = _run_foc(cfg, float(args.dt), float(args.t_end), seed=123)
        foc = _summarize(foc_series, float(args.window_frac))

        best = None
        for r in id_ratios:
            foc_cfg = replace(cfg.foc, id_ref=id_ref_base * r)
            cfg_r = replace(cfg, foc=foc_cfg)
            series = _run_foc(cfg_r, float(args.dt), float(args.t_end), seed=123)
            metrics = _summarize(series, float(args.window_frac))
            if metrics["mean_abs_speed_error"] > foc["mean_abs_speed_error"] * (1.0 + float(args.err_tol)):
                continue
            if best is None or metrics["mean_p_in_pos"] < best["mean_p_in_pos"]:
                best = {
                    "id_ref": id_ref_base * r,
                    "mean_p_in_pos": metrics["mean_p_in_pos"],
                    "err_ratio": metrics["mean_abs_speed_error"] / max(foc["mean_abs_speed_error"], 1e-9),
                }

        if best is None:
            best = {
                "id_ref": id_ref_base,
                "mean_p_in_pos": foc["mean_p_in_pos"],
                "err_ratio": 1.0,
            }

        saving = (foc["mean_p_in_pos"] - best["mean_p_in_pos"]) / max(foc["mean_p_in_pos"], 1e-9) * 100.0
        rows.append(
            {
                "load_pct": pct,
                "best_id_ref": f"{best['id_ref']:.4g}",
                "FOC_P_W": f"{foc['mean_p_in_pos']:.4g}",
                "BEST_P_W": f"{best['mean_p_in_pos']:.4g}",
                "saving_pct": f"{saving:.2f}",
                "err_ratio": f"{best['err_ratio']:.3f}",
            }
        )
        map_loads.append(float(load))
        map_ids.append(float(best["id_ref"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "flux_map_table.md"
    md_path.write_text(_format_table(rows), encoding="utf-8")
    map_payload = {
        "load_points": map_loads,
        "id_ref_points": map_ids,
        "base_load": base_load,
        "omega_pu": float(args.omega_pu),
    }
    (out_dir / "optimal_map.json").write_text(json.dumps(map_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[flux_map] saved: {md_path}")


if __name__ == "__main__":
    main()
