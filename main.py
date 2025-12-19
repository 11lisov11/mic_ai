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
    parser.add_argument(
        "--demo",
        choices=["one_experiment", "testsuite", "score_testsuite", "score_testsuite_full", "identify_only"],
        default=None,
        help="run a bench demo instead of the legacy simulator",
    )
    parser.add_argument(
        "--with-identification",
        action="store_true",
        help="run identification before testsuite/score",
    )
    parser.add_argument("--policy-id", default=None, help="policy identifier for scoring/leaderboard")
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
    if args.demo == "one_experiment":
        from bench.orchestrator import default_experiment_config, run_experiment

        bench_cfg = default_experiment_config(env_cfg)
        speed_profile = bench_cfg.speed_profile
        if speed_profile is not None:
            speed_profile = replace(speed_profile, step_time=0.1 * args.t_end)
        bench_cfg = replace(
            bench_cfg,
            dt=args.dt,
            duration=args.t_end,
            controller=args.mode,
            speed_profile=speed_profile,
        )
        result = run_experiment(bench_cfg, policy=None)
        print(f"Saved bench logs to {result.log_dir}")
        return result.log_dir if result.log_dir is not None else Path("logs")
    if args.demo == "testsuite":
        from bench.testsuite import TestsuiteConfig, run_testsuite

        duration = args.t_end if args.t_end != ENV.sim.t_end else 60.0
        suite_cfg = TestsuiteConfig(
            duration=duration,
            dt=args.dt,
            controller=args.mode,
            with_identification=args.with_identification,
        )
        result = run_testsuite(env=env_cfg, suite_cfg=suite_cfg)
        print(f"Saved testsuite logs to {result.run_dir}")
        return result.run_dir
    if args.demo == "score_testsuite":
        from bench.scoring import score_testsuite
        from bench.testsuite import TestsuiteConfig, run_testsuite

        duration = args.t_end if args.t_end != ENV.sim.t_end else 60.0
        suite_cfg = TestsuiteConfig(
            duration=duration,
            dt=args.dt,
            controller=args.mode,
            with_identification=args.with_identification,
        )
        suite_result = run_testsuite(env=env_cfg, suite_cfg=suite_cfg)
        policy_id = args.policy_id or f"{args.mode}_baseline"
        summary = score_testsuite(suite_result.run_dir, policy_id=policy_id)
        print(f"Score: {summary['score']:.4f}")
        print(f"Updated leaderboard at {summary['leaderboard_path']}")
        return suite_result.run_dir
    if args.demo == "score_testsuite_full":
        from bench.scoring import score_testsuite
        from bench.testsuite import TestsuiteConfig, run_testsuite

        duration = args.t_end if args.t_end != ENV.sim.t_end else 60.0
        suite_cfg = TestsuiteConfig(
            duration=duration,
            dt=args.dt,
            controller=args.mode,
            with_identification=True,
        )
        suite_result = run_testsuite(env=env_cfg, suite_cfg=suite_cfg)
        policy_id = args.policy_id or f"{args.mode}_baseline"
        summary = score_testsuite(suite_result.run_dir, policy_id=policy_id)
        print(f"Score: {summary['score']:.4f}")
        print(f"Updated leaderboard at {summary['leaderboard_path']}")
        return suite_result.run_dir
    if args.demo == "identify_only":
        from bench.identification import IdentificationConfig, run_id_sequence
        import bench.orchestrator as orchestrator

        duration = args.t_end if args.t_end != ENV.sim.t_end else 2.0
        id_cfg = IdentificationConfig(
            env=env_cfg,
            dt=args.dt,
            duration=duration,
            pulse_start=0.1 * duration,
            pulse_end=0.8 * duration,
        )
        result = run_id_sequence(orchestrator, id_cfg, base_policy=None)
        print(f"Identification aging: {result['aging']:.4f}")
        print(f"ID log: {result.get('id_log_path')}")
        return Path("logs")
    result_path = run_simulation(env_cfg)
    print(f"Saved results to {result_path}")
    if args.plot:
        from outputs.plots import plot_run  # local import to avoid hard dependency if plotting is off

        plot_run(str(result_path))
        print("Saved plots to outputs/figures")
    return result_path


if __name__ == "__main__":
    main()
