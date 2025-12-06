"""
Оценки параметров двигателя по отклику на ступенчатое воздействие.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .motor_params import MotorParamsEstimated

try:
    from scipy.optimize import least_squares
except Exception as exc:  # pragma: no cover - необязательная зависимость
    least_squares = None
    _SCIPY_IMPORT_ERROR = exc


def _validate_arrays(data: dict, keys: tuple[str, ...]) -> None:
    lengths = []
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in data")
        arr = np.asarray(data[key])
        if arr.size == 0:
            raise ValueError(f"Array '{key}' is empty")
        lengths.append(arr.size)
    if len(set(lengths)) != 1:
        raise ValueError("Input arrays must have the same length")


def _step_mask(u_d: np.ndarray) -> np.ndarray:
    u_max = float(np.max(u_d))
    if u_max <= 0:
        raise ValueError("u_d does not contain a positive step")
    mask = u_d > 0.5 * u_max
    if not np.any(mask):
        raise ValueError("Unable to detect voltage step interval")
    return mask


def estimate_rs_from_pulse(data: dict) -> Tuple[float, float]:
    """
    Оценить Rs по установившемуся току во время ступеньки напряжения.
    """
    _validate_arrays(data, ("t", "u_d", "i_d"))
    t = np.asarray(data["t"], dtype=float)
    u_d = np.asarray(data["u_d"], dtype=float)
    i_d = np.asarray(data["i_d"], dtype=float)

    mask = _step_mask(u_d)
    U_step = float(np.mean(u_d[mask]))

    idx = np.where(mask)[0]
    start, end = idx[0], idx[-1] + 1
    steady_start = start + int(0.7 * (end - start))
    steady_start = min(steady_start, end - 1)
    steady_slice = slice(steady_start, end)
    I_inf = float(np.mean(i_d[steady_slice]))
    if abs(I_inf) < 1e-9:
        raise ValueError("Steady-state current is too close to zero for Rs estimation")

    Rs_est = U_step / I_inf
    fit_error = float(np.std(i_d[steady_slice]) / abs(I_inf))
    return Rs_est, fit_error


def estimate_leq_from_dynamics(data: dict, Rs_est: float) -> Tuple[float, float]:
    """
    Оценить эквивалентную индуктивность по переходному процессу.
    """
    if Rs_est == 0:
        raise ValueError("Rs_est must be non-zero for Leq estimation")

    _validate_arrays(data, ("t", "u_d", "i_d"))
    t = np.asarray(data["t"], dtype=float)
    u_d = np.asarray(data["u_d"], dtype=float)
    i_d = np.asarray(data["i_d"], dtype=float)

    mask = _step_mask(u_d)
    t_step = t[mask]
    i_step = i_d[mask]
    if t_step.size < 5:
        raise ValueError("Not enough points on voltage step for Leq estimation")

    idx = np.where(mask)[0]
    start, end = idx[0], idx[-1] + 1
    steady_start = start + int(0.7 * (end - start))
    steady_start = min(steady_start, end - 1)
    I_inf = float(np.mean(i_d[steady_start:end]))
    if abs(I_inf) < 1e-9:
        raise ValueError("Steady-state current is too close to zero for Leq estimation")

    y = 1.0 - i_step / I_inf
    valid = y > 1e-6
    if not np.any(valid):
        raise ValueError("Transient current response does not allow Leq estimation (y <= 0)")

    t_rel = t_step[valid] - t_step[valid][0]
    ln_y = np.log(y[valid])

    t_mean = float(np.mean(t_rel))
    ln_mean = float(np.mean(ln_y))
    denom = float(np.sum((t_rel - t_mean) ** 2))
    if denom <= 0:
        raise ValueError("Time variance is zero during regression for Leq")
    a = float(np.sum((t_rel - t_mean) * (ln_y - ln_mean)) / denom)
    b = ln_mean - a * t_mean

    if a >= 0:
        raise ValueError("Estimated slope is non-negative; cannot derive time constant")

    tau = -1.0 / a
    Leq_est = tau * Rs_est
    ln_y_fit = a * t_rel + b
    fit_error = float(np.sqrt(np.mean((ln_y - ln_y_fit) ** 2)))
    return Leq_est, fit_error


def estimate_lm(data: dict, Rs_est: float, Leq_est: float) -> float:
    """
    Грубая оценка намагничивающей индуктивности на основе эквивалентной.
    """
    if Leq_est <= 0:
        raise ValueError("Leq_est must be positive to estimate Lm")
    return 0.9 * Leq_est


def refine_params_with_model(
    data: dict,
    initial_est: MotorParamsEstimated,
    env_model_factory: Callable[[MotorParamsEstimated], object],
    use_rr: bool = False,
) -> MotorParamsEstimated:
    """
    Уточнить Rs/Ls/Lr/Lm (и при необходимости Rr) через мировую модель и МНК по i_d(t).
    """
    if least_squares is None:
        raise ImportError(
            f"scipy.optimize.least_squares is required for refinement: {_SCIPY_IMPORT_ERROR}"
        )

    _validate_arrays(data, ("t", "u_d", "i_d"))
    t = np.asarray(data["t"], dtype=float)
    u_d = np.asarray(data["u_d"], dtype=float)
    i_d_meas = np.asarray(data["i_d"], dtype=float)

    if t.size != u_d.size or t.size != i_d_meas.size:
        raise ValueError("t, u_d, and i_d must have the same length")
    if t.size < 5:
        raise ValueError("Not enough samples for refinement")

    def _param_bounds(val: float | None) -> tuple[float, float, float]:
        if val is None or val <= 0:
            start = 0.5
        else:
            start = float(val)
        lower = max(1e-4, 0.1 * start)
        upper = 10.0 * start
        return start, lower, upper

    start_rs, low_rs, up_rs = _param_bounds(initial_est.Rs)
    start_ls, low_ls, up_ls = _param_bounds(initial_est.Ls)
    start_lr, low_lr, up_lr = _param_bounds(initial_est.Lr)
    start_lm, low_lm, up_lm = _param_bounds(initial_est.Lm)
    start_rr, low_rr, up_rr = _param_bounds(initial_est.Rr)

    if use_rr:
        x0 = np.array([start_rs, start_rr, start_ls, start_lr, start_lm], dtype=float)
        lower = np.array([low_rs, low_rr, low_ls, low_lr, low_lm], dtype=float)
        upper = np.array([up_rs, up_rr, up_ls, up_lr, up_lm], dtype=float)
    else:
        x0 = np.array([start_rs, start_ls, start_lr, start_lm], dtype=float)
        lower = np.array([low_rs, low_ls, low_lr, low_lm], dtype=float)
        upper = np.array([up_rs, up_ls, up_lr, up_lm], dtype=float)

    def _extract_id(env_model) -> float:
        if hasattr(env_model, "i_d"):
            return float(env_model.i_d)
        if hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "i_d"):
            return float(env_model.motor_state.i_d)
        if hasattr(env_model, "motor") and hasattr(env_model.motor, "_currents") and hasattr(env_model.motor, "state"):
            i_d_val, _, _, _ = env_model.motor._currents(env_model.motor.state)  # type: ignore[attr-defined]
            return float(i_d_val)
        # TODO: адаптировать под реальный API env_model для получения i_d
        raise ValueError("Cannot extract i_d from env_model; expose .i_d or motor currents.")

    def _apply_voltage(env_model, u_d_val: float, u_q_val: float = 0.0) -> None:
        if hasattr(env_model, "set_voltage_dq"):
            env_model.set_voltage_dq(u_d_val, u_q_val)
        elif hasattr(env_model, "base_controller") and hasattr(env_model.base_controller, "set_voltage_dq"):
            env_model.base_controller.set_voltage_dq(u_d_val, u_q_val)
        else:
            # TODO: адаптировать под реальный setter напряжения env_model
            pass

        if hasattr(env_model, "step"):
            try:
                env_model.step(u_d_val, u_q_val)
            except TypeError:
                env_model.step()
        else:
            raise ValueError("env_model does not implement step()")

    def _build_estimated(theta: np.ndarray) -> MotorParamsEstimated:
        if use_rr:
            Rs, Rr, Ls, Lr, Lm = theta
        else:
            Rs, Ls, Lr, Lm = theta
            Rr = initial_est.Rr
        return MotorParamsEstimated(
            Rs=float(Rs),
            Rr=float(Rr) if Rr is not None else None,
            Ls=float(Ls),
            Lr=float(Lr),
            Lm=float(Lm),
        )

    def residuals(theta: np.ndarray) -> np.ndarray:
        est_params = _build_estimated(theta)
        env_model = env_model_factory(est_params)
        if hasattr(env_model, "reset"):
            env_model.reset()

        i_model = np.zeros_like(i_d_meas)
        for idx, u in enumerate(u_d):
            _apply_voltage(env_model, float(u), 0.0)
            i_model[idx] = _extract_id(env_model)
        return i_model - i_d_meas

    result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf", max_nfev=120)
    theta_opt = result.x
    return _build_estimated(theta_opt)


def _param_bounds(val: float | None, fallback: float) -> tuple[float, float, float]:
    start = fallback if val is None or val <= 0 else float(val)
    lower = max(1e-6, 0.1 * start)
    upper = 10.0 * start
    return start, lower, upper


def _apply_voltage_generic(env_model, u_d: float, u_q: float = 0.0) -> None:
    if hasattr(env_model, "set_voltage_dq"):
        env_model.set_voltage_dq(u_d, u_q)
    elif hasattr(env_model, "base_controller") and hasattr(env_model.base_controller, "set_voltage_dq"):
        env_model.base_controller.set_voltage_dq(u_d, u_q)
    # TODO: адаптировать под проектный API, если установка напряжений отличается


def _step_env_generic(env_model, u_d: float, u_q: float = 0.0) -> None:
    _apply_voltage_generic(env_model, u_d, u_q)
    if hasattr(env_model, "step"):
        try:
            env_model.step(u_d, u_q)
        except TypeError:
            env_model.step()
    else:
        raise ValueError("env_model must implement step()")


def _extract_i_d(env_model) -> float:
    if hasattr(env_model, "i_d"):
        return float(env_model.i_d)
    if hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "i_d"):
        return float(env_model.motor_state.i_d)
    if hasattr(env_model, "motor") and hasattr(env_model.motor, "_currents") and hasattr(env_model.motor, "state"):
        i_d_val, _, _, _ = env_model.motor._currents(env_model.motor.state)  # type: ignore[attr-defined]
        return float(i_d_val)
    raise ValueError("Cannot extract i_d from env_model")


def _extract_i_q(env_model) -> float:
    if hasattr(env_model, "i_q"):
        return float(env_model.i_q)
    if hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "i_q"):
        return float(env_model.motor_state.i_q)
    if hasattr(env_model, "motor") and hasattr(env_model.motor, "_currents") and hasattr(env_model.motor, "state"):
        _, i_q_val, _, _ = env_model.motor._currents(env_model.motor.state)  # type: ignore[attr-defined]
        return float(i_q_val)
    raise ValueError("Cannot extract i_q from env_model")


def _extract_torque(env_model) -> float | None:
    if hasattr(env_model, "torque"):
        return float(env_model.torque)
    if hasattr(env_model, "torque_e"):
        return float(env_model.torque_e)
    if hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "torque"):
        return float(env_model.motor_state.torque)
    if hasattr(env_model, "last_torque"):
        return float(env_model.last_torque)
    return None


def _extract_omega(env_model) -> float:
    if hasattr(env_model, "omega_m"):
        return float(env_model.omega_m)
    if hasattr(env_model, "w_mech"):
        return float(env_model.w_mech)
    if hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "omega_m"):
        return float(env_model.motor_state.omega_m)
    if hasattr(env_model, "motor") and hasattr(env_model.motor, "state") and hasattr(env_model.motor.state, "omega_m"):
        return float(env_model.motor.state.omega_m)
    raise ValueError("Cannot extract omega from env_model")


def _simulate_rs_leq(env_model, data: dict) -> dict:
    if "u_d" not in data or "t" not in data:
        raise ValueError("rs_leq data must contain 't' and 'u_d'")
    u_d = np.asarray(data["u_d"], dtype=float)
    i_d_model = np.zeros_like(u_d)
    psi_d_model = np.zeros_like(u_d)

    if hasattr(env_model, "reset"):
        env_model.reset()

    for idx, u in enumerate(u_d):
        _step_env_generic(env_model, float(u), 0.0)
        i_d_model[idx] = _extract_i_d(env_model)
        # Дополнительно сохраняем поток, если доступен
        if hasattr(env_model, "psi_d"):
            psi_d_model[idx] = float(env_model.psi_d)
        elif hasattr(env_model, "motor_state") and hasattr(env_model.motor_state, "psi_ds"):
            psi_d_model[idx] = float(env_model.motor_state.psi_ds)
        else:
            psi_d_model[idx] = 0.0

    return {"i_d_model": i_d_model, "psi_d_model": psi_d_model}


def _simulate_locked_rotor_q(env_model, data: dict) -> dict:
    if "u_q" not in data or "t" not in data:
        raise ValueError("locked_rotor_q data must contain 't' and 'u_q'")
    u_q = np.asarray(data["u_q"], dtype=float)
    u_d = np.asarray(data.get("u_d", np.zeros_like(u_q)), dtype=float)
    i_q_model = np.zeros_like(u_q)
    torque_model = np.zeros_like(u_q)

    if hasattr(env_model, "reset"):
        env_model.reset()
    # Пытаемся эмулировать закреплённый ротор, принудительно обнуляя скорость на каждом шаге
    def _lock_rotor():
        if hasattr(env_model, "motor") and hasattr(env_model.motor, "state"):
            env_model.motor.state.omega_m = 0.0  # type: ignore[attr-defined]
        if hasattr(env_model, "omega_m"):
            try:
                env_model.omega_m = 0.0  # type: ignore[assignment]
            except Exception:
                pass

    for idx, (ud_val, uq_val) in enumerate(zip(u_d, u_q)):
        _lock_rotor()
        _step_env_generic(env_model, float(ud_val), float(uq_val))
        _lock_rotor()
        i_q_model[idx] = _extract_i_q(env_model)
        torque = _extract_torque(env_model)
        torque_model[idx] = torque if torque is not None else 0.0

    return {"i_q_model": i_q_model, "torque_model": torque_model}


def _simulate_mech_runup_coast(env_model, data: dict) -> dict:
    if "torque_cmd" not in data or "t" not in data:
        raise ValueError("mech_runup_coast data must contain 't' and 'torque_cmd'")
    torque_cmd = np.asarray(data["torque_cmd"], dtype=float)
    omega_model = np.zeros_like(torque_cmd)

    if hasattr(env_model, "reset"):
        env_model.reset()

    for idx, tq in enumerate(torque_cmd):
        # TODO: адаптировать команду момента под реальный API; здесь используем возбуждение по q-оси.
        _step_env_generic(env_model, 0.0, float(tq))
        omega_model[idx] = _extract_omega(env_model)

    return {"omega_model": omega_model}


def refine_params_multi_test(
    data_tests: dict,
    initial_est: MotorParamsEstimated,
    env_model_factory: Callable[[MotorParamsEstimated], object],
    use_rr: bool = True,
) -> MotorParamsEstimated:
    """
    Многотестовая оптимизация параметров двигателя (Rs, Rr?, Ls, Lr, Lm, J, B).
    """
    if least_squares is None:
        raise ImportError(
            f"scipy.optimize.least_squares is required for multi-test refinement: {_SCIPY_IMPORT_ERROR}"
        )

    data_rs = data_tests.get("rs_leq")
    data_locked = data_tests.get("locked_rotor_q")
    data_mech = data_tests.get("mech_runup_coast")

    if data_rs is None and data_locked is None and data_mech is None:
        raise ValueError("No test data provided for multi-test refinement")

    start_rs, low_rs, up_rs = _param_bounds(initial_est.Rs, 0.5)
    start_ls, low_ls, up_ls = _param_bounds(initial_est.Ls, 0.1)
    start_lr, low_lr, up_lr = _param_bounds(initial_est.Lr, 0.1)
    start_lm, low_lm, up_lm = _param_bounds(initial_est.Lm, 0.1)
    # Для Rr стартуем примерно вдвое выше Rs, если есть; иначе берём 5.0
    rr_fallback = 5.0
    if initial_est.Rs is not None and initial_est.Rs > 0:
        rr_fallback = max(2.0 * float(initial_est.Rs), 1.0)
    start_rr, low_rr, up_rr = _param_bounds(initial_est.Rr, rr_fallback)
    start_j, low_j, up_j = _param_bounds(initial_est.J, 0.01)
    start_b, low_b, up_b = _param_bounds(initial_est.B, 1e-3)

    if use_rr:
        # Связываем Lr с Ls (симметрия) для стабилизации оценки
        x0 = np.array([start_rs, start_rr, start_ls, start_lm, start_j, start_b], dtype=float)
        lower = np.array([low_rs, low_rr, low_ls, low_lm, low_j, low_b], dtype=float)
        upper = np.array([up_rs, up_rr, up_ls, up_lm, up_j, up_b], dtype=float)
    else:
        x0 = np.array([start_rs, start_ls, start_lm, start_j, start_b], dtype=float)
        lower = np.array([low_rs, low_ls, low_lm, low_j, low_b], dtype=float)
        upper = np.array([up_rs, up_ls, up_lm, up_j, up_b], dtype=float)

    def _build_estimated(theta: np.ndarray) -> MotorParamsEstimated:
        if use_rr:
            Rs, Rr, Ls, Lm, J, B = theta
            Lr = Ls  # фиксируем симметрию
        else:
            Rs, Ls, Lm, J, B = theta
            Lr = Ls  # фиксируем симметрию
            Rr = initial_est.Rr
        return MotorParamsEstimated(
            Rs=float(Rs),
            Rr=float(Rr) if Rr is not None else None,
            Ls=float(Ls),
            Lr=float(Lr),
            Lm=float(Lm),
            J=float(J),
            B=float(B),
        )

    def residuals(theta: np.ndarray) -> np.ndarray:
        est_params = _build_estimated(theta)
        residuals_list: list[np.ndarray] = []

        # Остатки для теста RS/LEQ
        if data_rs is not None:
            env_model = env_model_factory(est_params)
            sim = _simulate_rs_leq(env_model, data_rs)
            i_d_meas = np.asarray(data_rs["i_d"], dtype=float)
            i_d_model = sim["i_d_model"]
            if i_d_meas.shape != i_d_model.shape:
                raise ValueError("Shape mismatch in rs_leq i_d")
            residuals_list.append(1.0 * (i_d_model - i_d_meas))
            if "psi_d" in data_rs:
                psi_meas = np.asarray(data_rs["psi_d"], dtype=float)
                psi_model = sim.get("psi_d_model")
                if psi_model is not None and psi_meas.shape == psi_model.shape:
                    residuals_list.append(0.1 * (psi_model - psi_meas))

        # Остатки теста q-оси с заблокированным ротором
        if data_locked is not None:
            env_model = env_model_factory(est_params)
            sim = _simulate_locked_rotor_q(env_model, data_locked)
            i_q_meas = np.asarray(data_locked["i_q"], dtype=float)
            torque_meas = np.asarray(data_locked.get("torque", np.zeros_like(i_q_meas)), dtype=float)
            i_q_model = sim["i_q_model"]
            torque_model = sim["torque_model"]
            if i_q_meas.shape != i_q_model.shape:
                raise ValueError("Shape mismatch in locked_rotor_q i_q")
            residuals_list.append(1.0 * (i_q_model - i_q_meas))
            if torque_meas.shape == torque_model.shape and np.any(torque_meas):
                residuals_list.append(2.0 * (torque_model - torque_meas))
            # Поощряем симметрию Lr ~ Ls для стабилизации оценки
            if est_params.Lr is not None and est_params.Ls is not None:
                residuals_list.append(0.2 * np.array([est_params.Lr - est_params.Ls], dtype=float))

        # Остатки для механического разгона/выбега
        if data_mech is not None:
            env_model = env_model_factory(est_params)
            sim = _simulate_mech_runup_coast(env_model, data_mech)
            omega_meas = np.asarray(data_mech["omega"], dtype=float)
            omega_model = sim["omega_model"]
            if omega_meas.shape != omega_model.shape:
                raise ValueError("Shape mismatch in mech_runup_coast omega")
            residuals_list.append(1.0 * (omega_model - omega_meas))

        if not residuals_list:
            raise ValueError("No residuals accumulated; ensure tests provided")
        return np.concatenate(residuals_list)

    result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf", max_nfev=120)
    theta_opt = result.x
    return _build_estimated(theta_opt)


__all__ = [
    "estimate_rs_from_pulse",
    "estimate_leq_from_dynamics",
    "estimate_lm",
    "refine_params_with_model",
    "refine_params_multi_test",
]
