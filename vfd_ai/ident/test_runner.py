"""
Run identification-oriented tests on the provided environment.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .test_sequences import make_rs_leq_test_profile


def _lock_rotor(env) -> None:
    """
    Force mechanical speed to zero to emulate locked-rotor behavior.
    """
    if hasattr(env, "motor") and hasattr(env.motor, "state"):
        env.motor.state.omega_m = 0.0  # type: ignore[attr-defined]
    if hasattr(env, "omega_m"):
        try:
            env.omega_m = 0.0  # type: ignore[assignment]
        except Exception:
            pass


def _apply_voltage_command(env, u_d: float, u_q: float) -> None:
    """
    Try to push a dq-voltage reference into the environment.
    """
    applied = False
    if hasattr(env, "set_voltage_dq"):
        env.set_voltage_dq(u_d, u_q)
        applied = True
    if hasattr(env, "base_controller") and hasattr(env.base_controller, "set_voltage_dq"):
        env.base_controller.set_voltage_dq(u_d, u_q)
        applied = True
    if hasattr(env, "controller") and hasattr(env.controller, "set_voltage_dq"):
        env.controller.set_voltage_dq(u_d, u_q)
        applied = True
    if hasattr(env, "u_d_ref"):
        env.u_d_ref = u_d
        applied = True
    if hasattr(env, "u_q_ref"):
        env.u_q_ref = u_q
        applied = True
    # TODO: integrate with project-specific API to set dq voltages (e.g., env.driver.set_voltage_dq)
    if not applied:
        raise ValueError(
            "Unable to apply dq voltage reference to environment; please connect your env's setter in _apply_voltage_command."
        )


def _step_env(env, u_d: float, u_q: float):
    """
    Advance the environment by one step, trying the most flexible call signature first.
    """
    try:
        return env.step(u_d, u_q)
    except TypeError:
        # Fallback when step() expects no arguments and uses internal references.
        _apply_voltage_command(env, u_d, u_q)
        return env.step()


def _extract_currents(env) -> Tuple[float, float]:
    """
    Extract i_d/i_q from the environment or its motor model.
    """
    if hasattr(env, "i_d") and hasattr(env, "i_q"):
        return float(env.i_d), float(env.i_q)
    if hasattr(env, "motor_state"):
        state = env.motor_state
        if hasattr(state, "i_d") and hasattr(state, "i_q"):
            return float(state.i_d), float(state.i_q)
    if hasattr(env, "motor") and hasattr(env.motor, "_currents") and hasattr(env.motor, "state"):
        i_d, i_q, _, _ = env.motor._currents(env.motor.state)  # type: ignore[attr-defined]
        return float(i_d), float(i_q)
    # TODO: map your environment's API to expose dq currents
    raise ValueError("Unable to read i_d/i_q from environment; expose them as attributes or via motor model.")


def _extract_mech_speed(env) -> float:
    if hasattr(env, "omega_m"):
        return float(env.omega_m)
    if hasattr(env, "w_mech"):
        return float(env.w_mech)
    if hasattr(env, "motor_state") and hasattr(env.motor_state, "omega_m"):
        return float(env.motor_state.omega_m)
    if hasattr(env, "motor") and hasattr(env.motor, "state") and hasattr(env.motor.state, "omega_m"):
        return float(env.motor.state.omega_m)
    # TODO: map your environment's API to expose mechanical speed
    raise ValueError("Unable to read mechanical speed from environment.")


def _extract_torque(env) -> float | None:
    if hasattr(env, "torque"):
        return float(env.torque)
    if hasattr(env, "torque_e"):
        return float(env.torque_e)
    if hasattr(env, "motor_state") and hasattr(env.motor_state, "torque"):
        return float(env.motor_state.torque)
    if hasattr(env, "last_torque"):
        return float(env.last_torque)
    # TODO: map your environment's API to expose electromagnetic torque
    return None


def run_rs_leq_test(env, u_d_step: float, total_time: float) -> Tuple[dict, dict]:
    """
    Run a d-axis voltage step test to estimate Rs and equivalent inductance.

    Returns tuple (data, meta).
    """
    if not hasattr(env, "dt"):
        raise ValueError("Environment must expose dt attribute.")
    dt = float(env.dt)
    profile = make_rs_leq_test_profile(dt, total_time, u_d_step)

    if hasattr(env, "reset"):
        env.reset()

    t_series = profile["t"]
    u_d_ref = profile["u_d_ref"]
    u_q_ref = profile["u_q_ref"]

    logged_t = np.zeros_like(t_series)
    logged_u_d = np.zeros_like(t_series)
    logged_u_q = np.zeros_like(t_series)
    logged_i_d = np.zeros_like(t_series)
    logged_i_q = np.zeros_like(t_series)
    logged_w_mech = np.zeros_like(t_series)

    for idx, t in enumerate(t_series):
        logged_t[idx] = t
        logged_u_d[idx] = u_d_ref[idx]
        logged_u_q[idx] = u_q_ref[idx]

        _lock_rotor(env)
        _step_env(env, u_d_ref[idx], u_q_ref[idx])
        _lock_rotor(env)

        i_d, i_q = _extract_currents(env)
        w_mech = _extract_mech_speed(env)

        logged_i_d[idx] = i_d
        logged_i_q[idx] = i_q
        logged_w_mech[idx] = w_mech

    data = {
        "t": logged_t,
        "u_d": logged_u_d,
        "u_q": logged_u_q,
        "i_d": logged_i_d,
        "i_q": logged_i_q,
        "w_mech": logged_w_mech,
    }
    meta = {"u_d_step": u_d_step, "total_time": total_time}
    return data, meta


def run_locked_rotor_q_test(env, u_q_step: float, total_time: float) -> Tuple[dict, dict]:
    """
    Locked-rotor style q-axis step to target Rr/Lr identification.
    """
    if not hasattr(env, "dt"):
        raise ValueError("Environment must expose dt attribute.")
    dt = float(env.dt)
    n_steps = int(np.floor(total_time / dt))
    if n_steps < 2:
        raise ValueError("total_time must span at least two steps")

    if hasattr(env, "reset"):
        env.reset()

    u_d_ref = np.zeros(n_steps, dtype=float)
    u_q_ref = np.zeros(n_steps, dtype=float)
    warmup = min(max(10, int(0.02 * n_steps)), n_steps - 1)
    u_q_ref[warmup:] = u_q_step
    t_series = np.arange(n_steps, dtype=float) * dt

    logged_t = np.zeros_like(t_series)
    logged_u_d = np.zeros_like(t_series)
    logged_u_q = np.zeros_like(t_series)
    logged_i_d = np.zeros_like(t_series)
    logged_i_q = np.zeros_like(t_series)
    logged_w_mech = np.zeros_like(t_series)
    logged_torque = np.zeros_like(t_series)

    for idx, t in enumerate(t_series):
        logged_t[idx] = t
        logged_u_d[idx] = u_d_ref[idx]
        logged_u_q[idx] = u_q_ref[idx]

        _lock_rotor(env)
        _step_env(env, u_d_ref[idx], u_q_ref[idx])
        _lock_rotor(env)

        i_d, i_q = _extract_currents(env)
        w_mech = _extract_mech_speed(env)
        torque = _extract_torque(env)

        logged_i_d[idx] = i_d
        logged_i_q[idx] = i_q
        logged_w_mech[idx] = w_mech
        logged_torque[idx] = torque if torque is not None else 0.0

    data = {
        "t": logged_t,
        "u_d": logged_u_d,
        "u_q": logged_u_q,
        "i_d": logged_i_d,
        "i_q": logged_i_q,
        "w_mech": logged_w_mech,
        "torque": logged_torque,
    }
    meta = {"u_q_step": u_q_step, "total_time": total_time}
    return data, meta


def run_mech_runup_coast_test(env, torque_ref: float, runup_time: float, coast_time: float) -> Tuple[dict, dict]:
    """
    Mechanical test: accelerate with torque_ref then coast.
    """
    if not hasattr(env, "dt"):
        raise ValueError("Environment must expose dt attribute.")
    dt = float(env.dt)
    n_run = int(np.floor(runup_time / dt))
    n_coast = int(np.floor(coast_time / dt))
    n_steps = max(n_run + n_coast, 2)

    if hasattr(env, "reset"):
        env.reset()

    torque_cmd = np.zeros(n_steps, dtype=float)
    torque_cmd[:n_run] = torque_ref
    t_series = np.arange(n_steps, dtype=float) * dt

    logged_t = np.zeros_like(t_series)
    logged_omega = np.zeros_like(t_series)
    logged_torque_cmd = np.zeros_like(t_series)

    for idx, t in enumerate(t_series):
        logged_t[idx] = t
        logged_torque_cmd[idx] = torque_cmd[idx]
        # TODO: adapt to real env_model torque command API; here we approximate via q-axis voltage/current command.
        _step_env(env, 0.0, torque_cmd[idx])
        logged_omega[idx] = _extract_mech_speed(env)

    data = {"t": logged_t, "omega": logged_omega, "torque_cmd": logged_torque_cmd}
    meta = {"torque_ref": torque_ref, "runup_time": runup_time, "coast_time": coast_time}
    return data, meta


__all__ = ["run_rs_leq_test", "run_locked_rotor_q_test", "run_mech_runup_coast_test"]
