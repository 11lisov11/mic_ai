"""
CLI для полной многотестовой идентификации двигателя (электрической + механической) с опциональной самопроверкой.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.core.env import make_env_from_config
from mic_ai.ident.auto_id import run_full_identification, self_check_full_identification
from mic_ai.ident.ident_result import IdentificationResult
from mic_ai.ident.io import save_ident_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full multi-test identification of induction motor parameters")
    parser.add_argument("--env-config", required=False, help="Path to environment configuration (optional if data provided)")
    parser.add_argument("--rs-leq-data", required=False, help="Path to recorded d-axis step test data (.npz or .json)")
    parser.add_argument("--locked-data", required=False, help="Path to recorded locked-rotor q test data (.npz or .json)")
    parser.add_argument("--mech-data", required=False, help="Path to recorded mechanical runup/coast data (.npz or .json)")
    parser.add_argument("--output", required=True, help="Path to save JSON with identification results")
    parser.add_argument("--motor-name", default="motor", help="Motor name for metadata")
    parser.add_argument("--source", default="simulation", help='Identification source (default: "simulation")')
    parser.add_argument("--no-refine", action="store_true", help="Disable refinement via multi-test optimization")
    parser.add_argument("--self-check", action="store_true", help="Run self-check loop instead of a single identification")
    return parser.parse_args(argv)


def pretty_print_ident_result(result: IdentificationResult) -> None:
    print("Estimated params:")
    for k, v in result.estimated.as_dict().items():
        print(f"  {k}: {v}")
    if result.true_params:
        print("True params:")
        for k, v in result.true_params.__dict__.items():
            print(f"  {k}: {v}")
    if result.rel_error:
        print("Relative error (%):")
        for k, v in result.rel_error.items():
            print(f"  {k}: {v:.2f}")
    if "refinement" in result.tests_meta:
        ref = result.tests_meta["refinement"]
        status = ref.get("status", "unknown")
        print(f"Refinement: {status} (method={ref.get('method')}, used_rr={ref.get('used_rr', True)})")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.env_config is None and not (args.rs_leq_data and args.locked_data and args.mech_data):
        raise ValueError(
            "Provide --env-config to run tests or supply all three data files (--rs-leq-data, --locked-data, --mech-data)."
        )

    make_env_cb = None
    if args.env_config is not None:
        make_env_cb = lambda: make_env_from_config(args.env_config)

    if args.self_check:
        if make_env_cb is None:
            raise ValueError("Self-check requires --env-config to build environments.")
        self_check_full_identification(make_env_cb)
        return

    env = make_env_cb() if make_env_cb is not None else None

    data_rs = data_locked = data_mech = None
    if args.rs_leq_data or args.locked_data or args.mech_data:
        from mic_ai.ident.io import load_test_data

        data_rs = load_test_data(args.rs_leq_data) if args.rs_leq_data else None
        data_locked = load_test_data(args.locked_data) if args.locked_data else None
        data_mech = load_test_data(args.mech_data) if args.mech_data else None
    result = run_full_identification(
        env,
        motor_name=args.motor_name,
        source=args.source,
        enable_refine=(not args.no_refine) and env is not None,
        data_rs_leq=data_rs,
        data_locked_rotor_q=data_locked,
        data_mech_runup=data_mech,
    )
    pretty_print_ident_result(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ident_result(result, str(output_path))
        print(f"Saved identification result to {output_path}")


if __name__ == "__main__":
    main()
