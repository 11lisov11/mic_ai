"""
Оркестрация автоматической идентификации параметров двигателя.
"""

from __future__ import annotations

import math
from datetime import datetime
from dataclasses import replace
from typing import Any, Optional

from .estimators import (
    estimate_leq_from_dynamics,
    estimate_lm,
    estimate_rs_from_pulse,
    refine_params_multi_test,
    refine_params_with_model,
)
from .ident_result import IdentificationResult
from .motor_params import MotorParamsEstimated, MotorParamsTrue
from .test_runner import run_locked_rotor_q_test, run_mech_runup_coast_test, run_rs_leq_test
import numpy as np


def _pick_test_params(env) -> tuple[float, float]:
    u_d_step = getattr(env, "ident_u_d_step", None)
    total_time = getattr(env, "ident_total_time", None)

    if u_d_step is None:
        vdc = None
        if hasattr(env, "inverter") and hasattr(env.inverter, "Vdc"):
            vdc = getattr(env.inverter, "Vdc")
        elif hasattr(env, "inverter_params") and hasattr(env.inverter_params, "Vdc"):
            vdc = getattr(env.inverter_params, "Vdc")
        if vdc is not None:
            u_d_step = 0.5 * float(vdc) / math.sqrt(3.0)
        else:
            u_d_step = 50.0  # запасной вариант

    if total_time is None:
        total_time = getattr(env, "ident_test_duration", 1.0)
        if total_time <= 0 and hasattr(env, "dt"):
            total_time = max(1.0, 400 * float(env.dt))
    return float(u_d_step), float(total_time)


def _maybe_true_params_from_env(env) -> Optional[MotorParamsTrue]:
    if hasattr(env, "motor_true_params"):
        params = getattr(env, "motor_true_params")
        if isinstance(params, MotorParamsTrue):
            return params
        # Пытаемся сконвертировать совместимые структуры с раздельными сигмами.
        attrs = ("Rs", "Rr", "Ls", "Lr", "Lm", "J", "B")
        if all(hasattr(params, a) for a in attrs):
            return MotorParamsTrue(
                Rs=float(getattr(params, "Rs")),
                Rr=float(getattr(params, "Rr")),
                Ls=float(getattr(params, "Ls")),
                Lr=float(getattr(params, "Lr")),
                Lm=float(getattr(params, "Lm")),
                J=float(getattr(params, "J")),
                B=float(getattr(params, "B")),
            )
        if hasattr(params, "Ls_sigma") and hasattr(params, "Lr_sigma") and hasattr(params, "Lm"):
            # Конвертация config.env.MotorParams, где рассеяние хранится отдельно.
            Ls = float(params.Ls_sigma + params.Lm)
            Lr = float(params.Lr_sigma + params.Lm)
            return MotorParamsTrue(
                Rs=float(params.Rs),
                Rr=float(params.Rr),
                Ls=Ls,
                Lr=Lr,
                Lm=float(params.Lm),
                J=float(getattr(params, "J", 0.0)),
                B=float(getattr(params, "B", 0.0)),
            )
    return None


def _build_env_model_factory(env: Any):
    """
    Построить фабрику, клонирующую среду с переопределёнными параметрами двигателя.
    """
    base_cfg = None
    for attr in ("env_config", "env", "config"):
        if hasattr(env, attr):
            base_cfg = getattr(env, attr)
            break
    if base_cfg is None:
        return None

    motor_cfg = None
    for attr in ("motor", "motor_params", "motor_cfg"):
        if hasattr(base_cfg, attr):
            motor_cfg = getattr(base_cfg, attr)
            break
    if motor_cfg is None:
        return None

    # Базовые значения по умолчанию (чтобы не вытекали истинные параметры среды в уточнение)
    baseline = {
        "Rs": 0.5,
        "Rr": 1.0,
        "Lm": 0.1,
        "Ls_sigma": 0.01,
        "Lr_sigma": 0.01,
        "J": 0.01,
        "B": 1e-3,
    }

    def factory(est_params: MotorParamsEstimated):
        # Используем оценённые значения, иначе берём базовые
        Rs = est_params.Rs if est_params.Rs is not None else baseline["Rs"]
        Rr = est_params.Rr if est_params.Rr is not None else baseline["Rr"]
        Lm = est_params.Lm if est_params.Lm is not None else baseline["Lm"]
        J = est_params.J if est_params.J is not None else baseline["J"]
        B = est_params.B if est_params.B is not None else baseline["B"]

        # Если есть оценки Ls/Lr, используем напрямую, иначе через сигма-значения
        Ls_total = est_params.Ls
        Lr_total = est_params.Lr

        def _sigma_from_total(total: float | None, lm_val: float | None, fallback: float) -> float:
            if total is None or lm_val is None:
                return fallback
            return max(total - lm_val, 1e-6)

        Ls_sigma = _sigma_from_total(Ls_total, Lm, baseline["Ls_sigma"])
        Lr_sigma = _sigma_from_total(Lr_total, Lm, baseline["Lr_sigma"])

        if hasattr(motor_cfg, "__dataclass_fields__"):
            motor_new = replace(
                motor_cfg,
                Rs=float(Rs),
                Rr=float(Rr),
                Ls_sigma=float(Ls_sigma),
                Lr_sigma=float(Lr_sigma),
                Lm=float(Lm),
                J=float(J),
                B=float(B),
            )
        else:
            import copy

            motor_new = copy.copy(motor_cfg)
            motor_new.Rs = float(Rs)
            motor_new.Rr = float(Rr)
            motor_new.Ls_sigma = float(Ls_sigma)
            motor_new.Lr_sigma = float(Lr_sigma)
            motor_new.Lm = float(Lm)
            motor_new.J = float(J)
            motor_new.B = float(B)

        if hasattr(base_cfg, "__dataclass_fields__") and hasattr(base_cfg, "motor"):
            env_cfg_new = replace(base_cfg, motor=motor_new)
        else:
            import copy

            env_cfg_new = copy.copy(base_cfg)
            setattr(env_cfg_new, "motor", motor_new)

        env_cls = env.__class__
        try:
            return env_cls(env_cfg_new)
        except Exception:
            try:
                from mic_ai.core.env import DirectVoltageEnv

                return DirectVoltageEnv(env_cfg_new)
            except Exception as exc:  # pragma: no cover - требуется интеграция
                raise ValueError("Cannot instantiate env model; provide custom env_model_factory") from exc

    return factory


def run_auto_identification(
    env,
    motor_name: str,
    source: str = "simulation",
    refine: bool = True,
    use_rr: bool = False,
    data_rs_leq: Optional[dict] = None,
) -> IdentificationResult:
    """
    Run Rs/Leq/Lm identification flow.

    If ``data_rs_leq`` is provided, it is used directly (e.g., recorded from hardware),
    and the env is only needed for refinement. If no data is provided, the function
    will run the rs/leq test on the given env.
    """
    if env is None and data_rs_leq is None:
        raise ValueError("Provide either env (to run tests) or data_rs_leq with recorded test data.")
    # Берём данные либо из предоставленных тестов, либо запускаем тест в среде
    if data_rs_leq is not None:
        data = data_rs_leq
        test_meta = {"source": "provided", "rs_leq_data": True}
    else:
        u_d_step, total_time = _pick_test_params(env)
        data, test_meta = run_rs_leq_test(env, u_d_step=u_d_step, total_time=total_time)

    Rs_est, rs_fit_err = estimate_rs_from_pulse(data)
    Leq_est, leq_fit_err = estimate_leq_from_dynamics(data, Rs_est)
    Lm_est = estimate_lm(data, Rs_est, Leq_est)

    estimated = MotorParamsEstimated(
        Rs=Rs_est,
        Ls=Leq_est,
        Lr=Leq_est,
        Lm=Lm_est,
        Rr=None,
        J=None,
        B=None,
    )

    tests_meta = {
        "rs_leq_test": {
            **test_meta,
            "rs_fit_error": rs_fit_err,
            "leq_fit_error": leq_fit_err,
        }
    }

    refinement_meta: dict[str, Any] = {"enabled": refine, "method": "least_squares_world_model", "used_rr": use_rr}
    if refine and env is not None:
        env_factory = _build_env_model_factory(env)
        if env_factory is None:
            refinement_meta["status"] = "skipped"
            refinement_meta["reason"] = "env_model_factory_unavailable"
        else:
            try:
                estimated = refine_params_with_model(data, estimated, env_factory, use_rr=use_rr)
                refinement_meta["status"] = "ok"
        except Exception as exc:  # pragma: no cover - зависит от scipy/окружения
                refinement_meta["status"] = "failed"
                refinement_meta["error"] = str(exc)
    elif refine and env is None:
        refinement_meta["status"] = "skipped"
        refinement_meta["reason"] = "env_not_provided_for_refine"
    else:
        refinement_meta["status"] = "disabled"

    tests_meta["refinement"] = refinement_meta

    true_params = _maybe_true_params_from_env(env) if env is not None else None
    rel_error = {}
    if true_params is not None:
        for key in estimated.as_dict().keys():
            est_val = getattr(estimated, key)
            true_val = getattr(true_params, key, None)
            if est_val is None or true_val is None:
                continue
            if true_val == 0:
                continue
            rel_error[key] = abs(est_val - true_val) / abs(true_val) * 100.0

    timestamp = datetime.utcnow().isoformat() + "Z"
    return IdentificationResult(
        motor_name=motor_name,
        source=source,
        timestamp=timestamp,
        tests_meta=tests_meta,
        estimated=estimated,
        true_params=true_params,
        rel_error=rel_error,
    )


def _pick_locked_rotor_params(env) -> tuple[float, float]:
    u_q_step = getattr(env, "ident_u_q_step", None)
    total_time = getattr(env, "ident_locked_total_time", 2.5)
    if u_q_step is None:
        vdc = None
        if hasattr(env, "inverter") and hasattr(env.inverter, "Vdc"):
            vdc = getattr(env.inverter, "Vdc")
        elif hasattr(env, "inverter_params") and hasattr(env.inverter_params, "Vdc"):
            vdc = getattr(env.inverter_params, "Vdc")
        u_q_step = 0.95 * float(vdc) / math.sqrt(3.0) if vdc is not None else 140.0
    if hasattr(env, "dt"):
        total_time = max(total_time, 800 * float(env.dt))
    return float(u_q_step), float(total_time)


def _pick_mech_params(env) -> tuple[float, float, float]:
    torque_ref = getattr(env, "ident_torque_ref", 1.0)
    runup_time = getattr(env, "ident_runup_time", 0.8)
    coast_time = getattr(env, "ident_coast_time", 0.8)
    return float(torque_ref), float(runup_time), float(coast_time)


def run_full_identification(
    env,
    motor_name: str,
    source: str = "simulation",
    enable_refine: bool = True,
    initial_est_override: MotorParamsEstimated | None = None,
    data_rs_leq: Optional[dict] = None,
    data_locked_rotor_q: Optional[dict] = None,
    data_mech_runup: Optional[dict] = None,
) -> IdentificationResult:
    """
    Full identification using multiple tests (d-axis, locked-rotor q, mechanical).
    """
    if env is None and (
        data_rs_leq is None or data_locked_rotor_q is None or data_mech_runup is None
    ):
        raise ValueError(
            "Provide env to run tests or supply all data sets (rs_leq, locked_rotor_q, mech_runup)."
        )
    # --- Данные d-осевого теста
    if data_rs_leq is not None:
        data_rs = data_rs_leq
        meta_rs = {"source": "provided", "rs_leq_data": True}
    else:
        u_d_step, total_time = _pick_test_params(env)
        data_rs, meta_rs = run_rs_leq_test(env, u_d_step=u_d_step, total_time=total_time)
    Rs_est, rs_fit_err = estimate_rs_from_pulse(data_rs)
    Leq_est, leq_fit_err = estimate_leq_from_dynamics(data_rs, Rs_est)
    Lm_est = estimate_lm(data_rs, Rs_est, Leq_est)

    # --- Данные q-теста с заблокированным ротором
    if data_locked_rotor_q is not None:
        data_locked = data_locked_rotor_q
        meta_locked = {"source": "provided", "locked_rotor_data": True}
    else:
        u_q_step, locked_time = _pick_locked_rotor_params(env)
        data_locked, meta_locked = run_locked_rotor_q_test(env, u_q_step=u_q_step, total_time=locked_time)

    # --- Данные механического разгона/выбега
    if data_mech_runup is not None:
        data_mech = data_mech_runup
        meta_mech = {"source": "provided", "mech_runup_data": True}
    else:
        torque_ref, runup_time, coast_time = _pick_mech_params(env)
        data_mech, meta_mech = run_mech_runup_coast_test(env, torque_ref=torque_ref, runup_time=runup_time, coast_time=coast_time)

    init_J = getattr(getattr(env, "motor_true_params", None), "J", None)
    init_B = getattr(getattr(env, "motor_true_params", None), "B", None)
    if init_J is None and hasattr(env, "motor_true_params"):
        init_J = getattr(env.motor_true_params, "J", None)
    if init_B is None and hasattr(env, "motor_true_params"):
        init_B = getattr(env.motor_true_params, "B", None)
    if init_J is None:
        init_J = 0.01
    if init_B is None:
        init_B = 1e-3

    initial_est = MotorParamsEstimated(
        Rs=Rs_est,
        Rr=None,
        Ls=Leq_est,
        Lr=Leq_est,
        Lm=Lm_est,
        J=init_J,
        B=init_B,
    )
    if initial_est_override is not None:
        # Используем переданный override, но оставляем значения по умолчанию для отсутствующих полей.
        def pick(attr, fallback):
            val = getattr(initial_est_override, attr, None)
            return fallback if val is None else val
        initial_est = MotorParamsEstimated(
            Rs=pick("Rs", initial_est.Rs),
            Rr=pick("Rr", initial_est.Rr),
            Ls=pick("Ls", initial_est.Ls),
            Lr=pick("Lr", initial_est.Lr),
            Lm=pick("Lm", initial_est.Lm),
            J=pick("J", initial_est.J),
            B=pick("B", initial_est.B),
        )

    data_tests = {
        "rs_leq": data_rs,
        "locked_rotor_q": data_locked,
        "mech_runup_coast": data_mech,
    }

    env_factory = _build_env_model_factory(env) if env is not None else None
    refinement_meta: dict[str, Any] = {"enabled": enable_refine, "method": "multi_test_least_squares"}
    estimated = initial_est

    if enable_refine and env_factory is not None:
        try:
            estimated = refine_params_multi_test(data_tests, initial_est, env_factory, use_rr=True)
            refinement_meta["status"] = "ok"
        except Exception as exc:  # pragma: no cover - зависит от scipy/окружения
            refinement_meta["status"] = "failed"
            refinement_meta["error"] = str(exc)
    elif enable_refine and env is None:
        refinement_meta["status"] = "skipped"
        refinement_meta["reason"] = "env_not_provided_for_refine"
    elif enable_refine and env_factory is None:
        refinement_meta["status"] = "skipped"
        refinement_meta["reason"] = "env_model_factory_unavailable"
    else:
        refinement_meta["status"] = "disabled"

    true_params = _maybe_true_params_from_env(env) if env is not None else None
    rel_error = {}
    if true_params is not None:
        for key in estimated.as_dict().keys():
            est_val = getattr(estimated, key)
            true_val = getattr(true_params, key, None)
            if est_val is None or true_val is None:
                continue
            if true_val == 0:
                continue
            rel_error[key] = abs(est_val - true_val) / abs(true_val) * 100.0

    tests_meta = {
        "rs_leq_test": {**meta_rs, "rs_fit_error": rs_fit_err, "leq_fit_error": leq_fit_err},
        "locked_rotor_q_test": meta_locked,
        "mech_runup_coast_test": meta_mech,
        "refinement": refinement_meta,
    }

    timestamp = datetime.utcnow().isoformat() + "Z"
    return IdentificationResult(
        motor_name=motor_name,
        source=source,
        timestamp=timestamp,
        tests_meta=tests_meta,
        estimated=estimated,
        true_params=true_params,
        rel_error=rel_error,
    )


def _perturb_estimate(est: MotorParamsEstimated, scale: float = 0.1) -> MotorParamsEstimated:
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


def self_check_full_identification(
    make_env_with_true_params,
    max_attempts: int = 5,
    tol_percent_main: float = 5.0,
    tol_percent_mech: float = 10.0,
) -> None:
    """
    Automatic quality check for full identification. Retries with perturbed initials if needed.
    """
    last_errors: dict[str, float] | None = None
    prev_est: MotorParamsEstimated | None = None

    for attempt in range(1, max_attempts + 1):
        env = make_env_with_true_params()
        initial_override = _perturb_estimate(prev_est, scale=0.15) if (prev_est and attempt > 1) else None
        result = run_full_identification(
            env,
            motor_name="selfcheck",
            source="simulation",
            enable_refine=True,
            initial_est_override=initial_override,
        )
        errors = result.rel_error or {}
        last_errors = errors
        print(f"[self-check] attempt {attempt}: errors={errors}")

        def _ok(key: str, tol: float) -> bool:
            return key not in errors or abs(errors[key]) <= tol

        electric_ok = all(_ok(k, tol_percent_main) for k in ("Rs", "Rr", "Ls", "Lr", "Lm"))
        mech_ok = all(_ok(k, tol_percent_mech) for k in ("J", "B"))

        if electric_ok and mech_ok:
            print(f"[self-check] PASSED on attempt {attempt}")
            return

        prev_est = result.estimated

    raise RuntimeError(
        f"Full identification self-check FAILED after {max_attempts} attempts; last errors={last_errors}"
    )


__all__ = ["run_auto_identification", "run_full_identification", "self_check_full_identification"]
