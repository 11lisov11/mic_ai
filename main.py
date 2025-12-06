"""
Утилита командной строки для запуска симуляций и при необходимости построения графиков.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# ensure project root importable when run as a script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import ENV, EnvConfig, SimulationParams, create_default_env  # noqa: E402
from simulation.run_simulation import run_simulation  # noqa: E402


def build_env_from_args(args: argparse.Namespace) -> EnvConfig:
    default_env = create_default_env()
    sim = replace(
        default_env.sim,
        t_end=args.t_end,
        dt=args.dt,
        mode=args.mode,
        scenario_name=args.scenario,
        save_prefix=args.save_prefix,
        load_torque=args.load_torque,
    )
    return replace(default_env, sim=sim)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Use ENV just for showing defaults in help
    parser = argparse.ArgumentParser(description="Run induction motor simulation")
    parser.add_argument("--mode", choices=["scalar", "foc"], default=ENV.sim.mode, help="control mode")
    parser.add_argument("--scenario", default=ENV.sim.scenario_name, help="scenario name")
    parser.add_argument("--t-end", type=float, default=ENV.sim.t_end, help="simulation time (s)")
    parser.add_argument("--dt", type=float, default=ENV.sim.dt, help="simulation step (s)")
    parser.add_argument("--save-prefix", default=ENV.sim.save_prefix, help="results filename prefix")
    parser.add_argument("--load-torque", type=float, default=ENV.sim.load_torque, help="load torque (Nm)")
    parser.add_argument("--plot", dest="plot", action="store_true", help="generate plots after run")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="skip plotting")
    parser.set_defaults(plot=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    env_cfg = build_env_from_args(args)
    result_path = run_simulation(env_cfg)
    print(f"Saved results to {result_path}")
    if args.plot:
        from outputs.plots import plot_run  # local import to avoid hard dependency if plotting is off

        plot_run(str(result_path))
        print("Saved plots to outputs/figures")
    return result_path


if __name__ == "__main__":
    main()
