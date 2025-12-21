"""
Identification test runners using the unified signal interface.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .signal_interface import IdentSignalInterface
from .test_sequences import make_rs_leq_test_profile


def run_rs_leq_test(env, u_d_step: float, total_time: float) -> Tuple[dict, dict]:
    """Run d-axis step test for Rs/Leq estimation (locked rotor)."""
    iface = IdentSignalInterface(env)
    dt = iface.dt
    profile = make_rs_leq_test_profile(dt, total_time, u_d_step)

    iface.reset()

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

        iface.lock_rotor(True)
        iface.apply_voltage_step(u_d_ref[idx], u_q_ref[idx])
        iface.lock_rotor(True)

        i_d, i_q = iface.read_currents_dq()
        w_mech = iface.read_mech_speed()

        logged_i_d[idx] = i_d
        logged_i_q[idx] = i_q
        logged_w_mech[idx] = w_mech

    iface.lock_rotor(False)

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
    """Run locked-rotor q-axis step test for Rr/Lr estimation."""
    iface = IdentSignalInterface(env)
    dt = iface.dt
    n_steps = int(np.floor(total_time / dt))
    if n_steps < 2:
        raise ValueError("total_time must span at least two steps")

    iface.reset()

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

        iface.lock_rotor(True)
        iface.apply_voltage_step(u_d_ref[idx], u_q_ref[idx])
        iface.lock_rotor(True)

        i_d, i_q = iface.read_currents_dq()
        w_mech = iface.read_mech_speed()
        torque = iface.read_torque()

        logged_i_d[idx] = i_d
        logged_i_q[idx] = i_q
        logged_w_mech[idx] = w_mech
        logged_torque[idx] = torque if torque is not None else 0.0

    iface.lock_rotor(False)

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
    """Mechanical run-up under torque command followed by coast."""
    iface = IdentSignalInterface(env)
    dt = iface.dt
    n_run = int(np.floor(runup_time / dt))
    n_coast = int(np.floor(coast_time / dt))
    n_steps = max(n_run + n_coast, 2)

    iface.reset()

    torque_cmd = np.zeros(n_steps, dtype=float)
    torque_cmd[:n_run] = torque_ref
    t_series = np.arange(n_steps, dtype=float) * dt

    logged_t = np.zeros_like(t_series)
    logged_omega = np.zeros_like(t_series)
    logged_torque_cmd = np.zeros_like(t_series)

    for idx, t in enumerate(t_series):
        logged_t[idx] = t
        logged_torque_cmd[idx] = torque_cmd[idx]
        iface.apply_torque_step(torque_cmd[idx])
        logged_omega[idx] = iface.read_mech_speed()

    data = {"t": logged_t, "omega": logged_omega, "torque_cmd": logged_torque_cmd}
    meta = {
        "torque_ref": torque_ref,
        "runup_time": runup_time,
        "coast_time": coast_time,
        "torque_command_mode": iface.torque_command_mode or "unknown",
    }
    return data, meta


__all__ = ["run_rs_leq_test", "run_locked_rotor_q_test", "run_mech_runup_coast_test"]
