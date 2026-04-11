from __future__ import annotations

import math
from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate

_base = create_default_env()

NAMEPLATE_ONBOARD = {
    "P_n": 550,
    "U_ll": 380,
    "I_n": 1.7,
    "cos_phi_n": 0.74,
    "eta_n": 0.78,
    "f_n": 50,
    "p": 2,
    "n_rated": 1390,
    "connection": "Y",
    "J": 0.02,
}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_ONBOARD)
_motor = replace(
    _motor_est,
    Rs=5.367757963,
    Rr=15.8200942,
    Lm=1.173697368,
    Ls_sigma=0.05,
    Lr_sigma=0.05,
    J=0.02,
    B=0.002196469545,
    p=2,
    I_n=1.7,
)

_sim = replace(
    _base.sim,
    t_end=2,
    dt=0.001,
    save_prefix="onboard_motor_x",
    scenario_name="speed_step",
    load_torque=0.9446246622,
)

_foc = replace(
    _base.foc,
    id_ref=0.595,
    iq_limit=5.1,
)

_inverter = replace(
    _base.inverter,
    Vdc=540,
    r_out=0.1,
    dead_time=2e-06,
    v_drop=1.2,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

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

# Filled by onboarding pipeline after training if desired.
ai_eval_checkpoint_path = ""
ai_eval_id_ref_alpha = 0.09675
ai_eval_delta_id_max = 0.108
ai_eval_id_ref_relative = True
ai_eval_id_ref_gate_speed_tol_rel = 0.08

__all__ = ["ENV"]
