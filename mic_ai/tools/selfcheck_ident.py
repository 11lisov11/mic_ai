"""
CLI для запуска самопроверки полной идентификации на конфигурации с известными параметрами.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.core.env import make_env_from_config
from mic_ai.ident.selfcheck import self_check_full_identification


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-check of full motor identification")
    parser.add_argument("--env-config", required=True, help="Path to config with true motor parameters (ENV)")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum identification attempts")
    parser.add_argument("--tol-main", type=float, default=5.0, help="Tolerance %% for Rs/Rr/Ls/Lr/Lm")
    parser.add_argument("--tol-mech", type=float, default=15.0, help="Tolerance %% for J/B")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def make_env():
        return make_env_from_config(args.env_config)

    try:
        self_check_full_identification(
            make_env_with_true_params=make_env,
            max_attempts=args.max_attempts,
            tol_percent_main=args.tol_main,
            tol_percent_mech=args.tol_mech,
        )
    except Exception as exc:  # pragma: no cover - путь завершения CLI
        print(f"[self-check] FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
