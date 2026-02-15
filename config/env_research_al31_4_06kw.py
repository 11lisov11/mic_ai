from __future__ import annotations

import math
from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate


_base = create_default_env()

NAMEPLATE_AL31_4_06KW = {
    "P_n": 0.60 * 1000.0,  # W
    "U_ll": 380.0,  # V
    "I_n": 1.70,  # A
    "cos_phi_n": 0.74,
    "eta_n": 0.78,
    "f_n": 50.0,  # Hz
    "p": 2,  # pole pairs (4-pole)
    "n_rated": 1390.0,  # rpm
    "connection": "Y",
    "J": 0.020,
}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_AL31_4_06KW)
_motor = replace(
    _motor_est,
    Ls_sigma=float(max(getattr(_motor_est, "Ls_sigma", 0.05), 0.045)),
    Lr_sigma=float(max(getattr(_motor_est, "Lr_sigma", 0.05), 0.045)),
    J=float(NAMEPLATE_AL31_4_06KW["J"]),
)

_torque_nom = float(NAMEPLATE_AL31_4_06KW["P_n"]) / max(
    2.0 * math.pi * float(NAMEPLATE_AL31_4_06KW["n_rated"]) / 60.0,
    1e-6,
)
_load_torque = 0.25 * _torque_nom

_sim = replace(
    _base.sim,
    t_end=2.0,
    # dt=1e-3 is numerically stable for this motor and keeps runtime reasonable.
    dt=1e-3,
    save_prefix="research_al31_4_06kw",
    scenario_name="speed_step",
    load_torque=float(_load_torque),
)

_foc = replace(
    _base.foc,
    id_ref=1.50,
    iq_limit=5.2,
)

_inverter = replace(
    _base.inverter,
    r_out=0.10,
    dead_time=2e-6,
    v_drop=1.2,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

loss_inv_r = 1.650000
loss_core_k = 0.180000
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

id_ref_lut_path = None

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
