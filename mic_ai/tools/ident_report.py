"""
Запустить идентификацию и вывести таблицу из 4 колонок: param | true | est | err%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.core.env import make_env_from_config
from mic_ai.ident.auto_id import run_auto_identification, run_full_identification
from mic_ai.ident.ident_result import IdentificationResult
from mic_ai.ident.motor_params import MotorParamsEstimated


def _jitter_estimate(est: MotorParamsEstimated, scale: float = 0.1) -> MotorParamsEstimated:
    def jitter(val: float | None, fallback: float) -> float:
        base = fallback if val is None else float(val)
        factor = 1.0 + float(np.random.normal(0.0, scale))
        return max(base * factor, 1e-6)

    return MotorParamsEstimated(
        Rs=jitter(est.Rs, 0.5),
        Rr=jitter(est.Rr, 1.0) if est.Rr is not None else None,
        Ls=jitter(est.Ls, 0.1),
        Lr=jitter(est.Lr, 0.1),
        Lm=jitter(est.Lm, 0.1),
        J=jitter(est.J, 0.01),
        B=jitter(est.B, 1e-3),
    )


def _relative_error(true_val: float, est_val: float) -> float:
    if true_val == 0:
        return 0.0
    return abs(est_val - true_val) / abs(true_val) * 100.0


def _collect_rows(result: IdentificationResult) -> List[Tuple[str, str, str, str]]:
    est = result.estimated.as_dict()
    true_params = result.true_params.__dict__ if result.true_params else {}
    rel = result.rel_error or {}

    # Предпочитаем параметры с известными истинными значениями; иначе показываем все оценённые
    keys = [k for k in est.keys() if est[k] is not None and k in true_params]
    if not keys:
        keys = [k for k in est.keys() if est[k] is not None]

    rows: List[Tuple[str, str, str, str]] = []
    for k in sorted(keys):
        est_val = est.get(k)
        true_val = true_params.get(k)
        if est_val is None:
            continue
        err_val: float | None = None
        if true_val is not None:
            err_val = rel.get(k, _relative_error(float(true_val), float(est_val)))
        rows.append(
            (
                k,
                "" if true_val is None else f"{float(true_val):.6g}",
                f"{float(est_val):.6g}",
                "" if err_val is None else f"{err_val:.3f}",
            )
        )
    return rows


def _score_result(result: IdentificationResult) -> float:
    rows = _collect_rows(result)
    errs: List[float] = []
    for _, _, _, err_str in rows:
        if err_str:
            errs.append(abs(float(err_str)))
    if not errs:
        return float("inf")
    return float(np.mean(errs))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run motor identification and print param table")
    parser.add_argument("--env-config", required=True, help="Path to config with ENV")
    parser.add_argument("--mode", choices=["full", "auto"], default="full", help="Identification mode")
    parser.add_argument("--no-refine", action="store_true", help="Disable refinement stage")
    return parser.parse_args(argv)


def _run_ident(make_env: Callable[[], object], mode: str, refine: bool):
    env = make_env()

    if mode == "auto":
        return run_auto_identification(env, motor_name="report", source="simulation", refine=refine, data_rs_leq=None)
    return run_full_identification(
        env,
        motor_name="report",
        source="simulation",
        enable_refine=refine,
        initial_est_override=None,
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    make_env = lambda: make_env_from_config(args.env_config)

    result = _run_ident(make_env, args.mode, refine=not args.no_refine)
    rows = _collect_rows(result)

    print(f"{'param':<8} {'true':>14} {'est':>14} {'err%':>8}")
    print("-" * 48)
    for name, true_str, est_str, err_str in rows:
        print(f"{name:<8} {true_str:>14} {est_str:>14} {err_str:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
