from __future__ import annotations

import math
from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate


_base = create_default_env()

NAMEPLATE_AIR56_025KW = {
    "P_n": 0.25 * 1000.0,  # W
    "U_ll": 380.0,  # V
    "I_n": 0.70,  # A
    "cos_phi_n": 0.68,
    "eta_n": 0.75,
    "f_n": 50.0,  # Hz
    "p": 2,  # pole pairs (4-pole)
    "n_rated": 1380.0,  # rpm
    "connection": "Y",
    "J": 0.010,
}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_AIR56_025KW)
_motor = replace(
    _motor_est,
    # Keep sigma inductances in a numerically robust band.
    Ls_sigma=float(max(getattr(_motor_est, "Ls_sigma", 0.05), 0.05)),
    Lr_sigma=float(max(getattr(_motor_est, "Lr_sigma", 0.05), 0.05)),
    J=float(NAMEPLATE_AIR56_025KW["J"]),
)

_torque_nom = float(NAMEPLATE_AIR56_025KW["P_n"]) / max(
    2.0 * math.pi * float(NAMEPLATE_AIR56_025KW["n_rated"]) / 60.0,
    1e-6,
)
_load_torque = 0.25 * _torque_nom

_sim = replace(
    _base.sim,
    t_end=2.0,
    # NOTE: dt=1e-3 produced numerical artefacts (spurious negative p_el in steady windows)
    # for this motor, which inflated Pвх+ metrics. We use a smaller dt for research figures.
    dt=5e-4,
    save_prefix="research_air56_025kw",
    scenario_name="speed_step",
    load_torque=float(_load_torque),
)

_foc = replace(
    _base.foc,
    # Fixed-flux baseline; MIC learns a better scheduling from data only.
    id_ref=1.35,
    iq_limit=2.2,
)

_inverter = replace(
    _base.inverter,
    r_out=0.12,
    dead_time=2e-6,
    v_drop=1.2,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

# Loss model used when scenario_compare is called with --use-total-power.
loss_inv_r = 3.706943
loss_core_k = 0.090869
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

# Force strict FOC-vs-MIC comparison without external flux map assistance.
id_ref_lut_path = None

# Domain randomization used for motor-agnostic RL training.
ai_omega_ref_pu_range = (0.3, 1.1)
ai_load_mult_range = (0.5, 1.6)
ai_drift_every_episodes = 1
ai_drift_params = ("Rs", "Rr", "Lm", "Ls_sigma", "Lr_sigma", "J", "B")
ai_drift_ranges = {
    "Rs": (0.75, 1.25),
    "Rr": (0.75, 1.25),
    "Lm": (0.85, 1.15),
    "Ls_sigma": (0.75, 1.25),
    "Lr_sigma": (0.75, 1.25),
    "J": (0.6, 1.4),
    "B": (0.6, 1.8),
}

__all__ = ["ENV"]
