"""
Помощник для самопроверки полной идентификации на известном двигателе.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .auto_id import run_full_identification
from .ident_result import IdentificationResult
from .motor_params import MotorParamsEstimated, MotorParamsTrue


MAIN_KEYS = ["Rs", "Rr", "Ls", "Lr", "Lm"]
MECH_KEYS = ["J", "B"]


def _jitter_estimate(est: MotorParamsEstimated, scale: float = 0.1) -> MotorParamsEstimated:
    """Возбудить предыдущую оценку, чтобы выйти из локальных минимумов между попытками."""

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


def _build_report(result: IdentificationResult) -> List[Tuple[str, float, float, float]]:
    """
    Вернуть список кортежей (имя, true, est, err%).
    Берём только параметры, присутствующие и в оценке, и в истинном наборе.
    """
    report: List[Tuple[str, float, float, float]] = []
    est_dict = result.estimated.as_dict()
    true_params: MotorParamsTrue | None = result.true_params
    rel_err: Dict[str, float] = result.rel_error or {}
    if true_params is None:
        return report
    true_dict = true_params.__dict__

    for key, est_val in est_dict.items():
        if est_val is None:
            continue
        if key not in true_dict:
            continue
        true_val = true_dict[key]
        if true_val is None:
            continue
        # Предпочитаем уже посчитанную относительную ошибку; иначе считаем вручную
        err = rel_err.get(key)
        if err is None:
            err = _relative_error(float(true_val), float(est_val))
        report.append((key, float(true_val), float(est_val), float(err)))
    return report


def self_check_full_identification(
    make_env_with_true_params: Callable[[], object],
    max_attempts: int = 5,
    tol_percent_main: float = 5.0,
    tol_percent_mech: float = 15.0,
) -> None:
    """
    Самопроверка: выполняет полную идентификацию на среде с известными параметрами.

    Пример:
        # make_env_from_config("config/env_demo_true.yaml")
        # (нужен модуль с ENV, содержащим true_params)
    """
    prev_est: MotorParamsEstimated | None = None
    last_report: List[Tuple[str, float, float, float]] = []

    for attempt in range(1, max_attempts + 1):
        env = make_env_with_true_params()
        initial_override = _jitter_estimate(prev_est, scale=0.15) if (prev_est and attempt > 1) else None

        result = run_full_identification(
            env,
            motor_name="selfcheck",
            source="simulation",
            enable_refine=True,
            initial_est_override=initial_override,
        )

        report = _build_report(result)
        last_report = report
        if not report:
            raise ValueError("Self-check requires env with true_params to compute errors.")

        fail_items = []
        for name, true_val, est_val, err in report:
            if name in MAIN_KEYS and abs(err) > tol_percent_main:
                fail_items.append((name, true_val, est_val, err, tol_percent_main))
            if name in MECH_KEYS and abs(err) > tol_percent_mech:
                fail_items.append((name, true_val, est_val, err, tol_percent_mech))

        print(f"[self-check] Attempt {attempt}/{max_attempts}")
        for name, true_val, est_val, err in report:
            print(f"  {name:>2}: true={true_val:.6g}, est={est_val:.6g}, err={err:.3f}%")

        if not fail_items:
            print(f"[self-check] SELF-CHECK PASSED on attempt {attempt}")
            return

        print(f"[self-check] FAILED on attempt {attempt}")
        for name, true_val, est_val, err, tol in fail_items:
            print(f"  param={name} err={err:.3f}% tol={tol}% (true={true_val:.6g}, est={est_val:.6g})")

        prev_est = result.estimated
        print("[self-check] retrying with perturbed initials...")

    print("[self-check] final report after max attempts:")
    for name, true_val, est_val, err in last_report:
        print(f"  {name:>2}: true={true_val:.6g}, est={est_val:.6g}, err={err:.3f}%")
    raise RuntimeError(
        f"Full identification self-check FAILED: relative error too high after {max_attempts} attempts."
    )


__all__ = ["self_check_full_identification"]
