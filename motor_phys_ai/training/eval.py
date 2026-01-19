# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from dataclasses import replace

from config.env import EnvConfig
from motor_phys_ai.controllers.physmod import PhysModController
from motor_phys_ai.controllers.pi import PIController
from motor_phys_ai.env.motor_env import MotorEnv
from motor_phys_ai.utils.metrics import calc_metrics, weighted_score
from motor_phys_ai.utils.plots import plot_timeseries


def _load_env_config(path: Path) -> EnvConfig:
    spec = importlib.util.spec_from_file_location("motor_phys_ai_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ENV"):
        raise AttributeError("Config module must define ENV")
    return getattr(module, "ENV")


def _save_csv(path: Path, series: Dict[str, np.ndarray]) -> None:
    keys = list(series.keys())
    if not keys:
        return
    n = int(series[keys[0]].size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for k in range(n):
            row = ",".join(f"{float(series[key][k])}" for key in keys)
            f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PhysMod vs PI controllers.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1_nominal.py")
    parser.add_argument("--scenarios", default="step,load,drift")
    parser.add_argument("--out-dir", default="motor_phys_ai/outputs/runs_phys")
    parser.add_argument("--run-tag", default="run_phys")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--speed-tol-rel", type=float, default=0.05)

    parser.add_argument("--pi-kp", type=float, default=0.6)
    parser.add_argument("--pi-ki", type=float, default=3.0)

    parser.add_argument("--phys-kp", type=float, default=0.6)
    parser.add_argument("--phys-ki", type=float, default=3.0)
    parser.add_argument("--phys-kt-init", type=float, default=1.0)
    parser.add_argument("--phys-kt-alpha", type=float, default=0.02)
    parser.add_argument("--phys-omega-dot-thr", type=float, default=1.0)

    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=0.1)
    parser.add_argument("--w-stability", type=float, default=5.0)
    args = parser.parse_args()

    env_path = Path(args.env_config).resolve()
    env_cfg = _load_env_config(env_path)
    if args.dt is not None:
        env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, dt=float(args.dt)))
    if args.t_end is not None:
        env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, t_end=float(args.t_end)))

    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    run_dir = Path(args.out_dir) / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    env = MotorEnv(env_cfg)
    iq_limit = float(getattr(env_cfg.foc, "iq_limit", 0.0) or 0.0)
    inertia_j = float(getattr(env_cfg.motor, "J", 0.0) or 0.0)

    controllers = {
        "PI": PIController(args.pi_kp, args.pi_ki, env.dt, iq_limit=iq_limit),
        "PhysMod": PhysModController(
            args.phys_kp,
            args.phys_ki,
            env.dt,
            inertia_j,
            args.phys_kt_init,
            iq_limit=iq_limit,
            kt_alpha=args.phys_kt_alpha,
            omega_dot_threshold=args.phys_omega_dot_thr,
        ),
    }

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario in scenarios:
        summary[scenario] = {}
        for name, ctrl in controllers.items():
            series = env.run(ctrl, scenario)
            metrics = calc_metrics(
                series["t"],
                series["omega"],
                series["omega_ref"],
                i_q=series["i_q"],
                iq_limit=iq_limit if iq_limit > 0 else None,
                speed_tol_rel=float(args.speed_tol_rel),
            )
            p_el = np.maximum(series.get("p_el", np.zeros_like(series["t"])), 0.0)
            p_el_mean = float(np.mean(p_el)) if p_el.size else 0.0
            score = weighted_score(metrics, args.w_speed, args.w_energy, args.w_stability)
            metrics["score"] = score
            metrics["p_el_mean"] = p_el_mean
            summary[scenario][name] = metrics

            csv_path = run_dir / f"{scenario}_{name.lower()}.csv"
            _save_csv(csv_path, series)
            plot_path = run_dir / f"{scenario}_{name.lower()}.pdf"
            plot_timeseries(series, plot_path, title=f"{scenario} / {name}")

    meta = {
        "env_config": str(env_path),
        "scenarios": scenarios,
        "weights": {"speed": args.w_speed, "energy": args.w_energy, "stability": args.w_stability},
        "speed_tol_rel": float(args.speed_tol_rel),
    }
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=2), encoding="utf-8")
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()
