"""
CLI-обёртка для автоматической идентификации параметров двигателя.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ident.auto_id import run_auto_identification
from mic_ai.ident.ident_result import IdentificationResult
from mic_ai.ident.io import save_ident_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify induction motor parameters via simulation tests")
    parser.add_argument("--env-config", required=False, help="Path to environment configuration (optional if data provided)")
    parser.add_argument("--rs-leq-data", required=False, help="Path to recorded d-axis step test data (.npz or .json)")
    parser.add_argument("--output", required=True, help="Path to save JSON with identification results")
    parser.add_argument("--motor-name", default="motor", help="Motor name for metadata")
    parser.add_argument("--source", default="simulation", help='Identification source (default: "simulation")')
    parser.add_argument("--no-refine", action="store_true", help="Disable refinement via world-model optimization")
    return parser.parse_args(argv)


def pretty_print_ident_result(result: IdentificationResult) -> None:
    print("Estimated params:")
    for k, v in result.estimated.as_dict().items():
        print(f"  {k}: {v}")
    if result.true_params:
        print("True params:")
        true_dict = result.true_params.__dict__
        for k, v in true_dict.items():
            print(f"  {k}: {v}")
    if result.rel_error:
        print("Relative error (%):")
        for k, v in result.rel_error.items():
            print(f"  {k}: {v:.2f}")
    if "refinement" in result.tests_meta:
        ref = result.tests_meta["refinement"]
        status = ref.get("status", "unknown")
        print(f"Refinement: {status} (method={ref.get('method')}, used_rr={ref.get('used_rr')})")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.env_config is None and args.rs_leq_data is None:
        raise ValueError("Provide either --env-config to run tests or --rs-leq-data with recorded test data.")

    env = None
    if args.env_config is not None:
        try:
            from mic_ai.core.env import make_env_from_config  # type: ignore
        except Exception as exc:  # pragma: no cover - требуется проектно-специфичная фабрика
            raise ImportError(
                "Expected make_env_from_config in mic_ai.core.env to build the digital twin environment."
            ) from exc
        env = make_env_from_config(args.env_config)

    data_rs = None
    if args.rs_leq_data:
        from mic_ai.ident.io import load_test_data

        data_rs = load_test_data(args.rs_leq_data)

    result = run_auto_identification(
        env,
        motor_name=args.motor_name,
        source=args.source,
        refine=(not args.no_refine) and env is not None,
        data_rs_leq=data_rs,
    )
    pretty_print_ident_result(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ident_result(result, str(output_path))
    print(f"Saved identification result to {output_path}")


if __name__ == "__main__":
    main()
