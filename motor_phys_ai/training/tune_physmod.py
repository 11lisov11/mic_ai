# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from config.env import EnvConfig
from motor_phys_ai.controllers.physmod import PhysModController
from motor_phys_ai.env.motor_env import MotorEnv
from motor_phys_ai.utils.metrics import calc_metrics, weighted_score


def _load_env_config(path: Path) -> EnvConfig:
    spec = importlib.util.spec_from_file_location("motor_phys_ai_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ENV"):
        raise AttributeError("Config module must define ENV")
    return getattr(module, "ENV")


def _parse_list(text: str) -> List[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for PhysMod parameters.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1_nominal.py")
    parser.add_argument("--scenarios", default="step,load,drift")
    parser.add_argument("--out-dir", default="motor_phys_ai/outputs/runs_phys")
    parser.add_argument("--run-tag", default="tune_phys")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--speed-tol-rel", type=float, default=0.05)

    parser.add_argument("--kp-grid", default="0.4,0.6,0.8")
    parser.add_argument("--ki-grid", default="2.0,3.0,4.0")
    parser.add_argument("--kt-grid", default="0.6,0.8,1.0,1.2,1.5")
    parser.add_argument("--kt-alpha", type=float, default=0.02)
    parser.add_argument("--omega-dot-thr", type=float, default=1.0)

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
    env = MotorEnv(env_cfg)
    iq_limit = float(getattr(env_cfg.foc, "iq_limit", 0.0) or 0.0)
    inertia_j = float(getattr(env_cfg.motor, "J", 0.0) or 0.0)

    kp_list = _parse_list(args.kp_grid)
    ki_list = _parse_list(args.ki_grid)
    kt_list = _parse_list(args.kt_grid)

    best_score = None
    best_params: Dict[str, float] = {}
    results: List[Dict[str, float]] = []

    for kp in kp_list:
        for ki in ki_list:
            for kt in kt_list:
                ctrl = PhysModController(
                    kp,
                    ki,
                    env.dt,
                    inertia_j,
                    kt,
                    iq_limit=iq_limit if iq_limit > 0 else None,
                    kt_alpha=args.kt_alpha,
                    omega_dot_threshold=args.omega_dot_thr,
                )
                total = 0.0
                for scenario in scenarios:
                    series = env.run(ctrl, scenario)
                    metrics = calc_metrics(
                        series["t"],
                        series["omega"],
                        series["omega_ref"],
                        i_q=series["i_q"],
                        iq_limit=iq_limit if iq_limit > 0 else None,
                        speed_tol_rel=float(args.speed_tol_rel),
                    )
                    score = weighted_score(metrics, args.w_speed, args.w_energy, args.w_stability)
                    total += score
                avg_score = total / max(len(scenarios), 1)
                results.append({"kp": kp, "ki": ki, "kt_init": kt, "score": avg_score})
                if best_score is None or avg_score < best_score:
                    best_score = avg_score
                    best_params = {"kp": kp, "ki": ki, "kt_init": kt}

    run_dir = Path(args.out_dir) / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (run_dir / "tune_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best params: {best_params}, score={best_score}")
    print(f"Saved tuning results to {run_dir}")


if __name__ == "__main__":
    main()
