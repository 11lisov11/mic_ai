from __future__ import annotations

import math
from dataclasses import replace

from config.env import create_default_env, estimate_id_ref_from_nameplate, estimate_motor_params_from_nameplate


_base = create_default_env()

NAMEPLATE_AO2_32_4_3KW = {
    "P_n": 3.00 * 1000.0,
    "U_ll": 380.0,
    "I_n": 7.20,
    "cos_phi_n": 0.82,
    "eta_n": 0.84,
    "f_n": 50.0,
    "p": 2,
    "n_rated": 1430.0,
    "connection": "Y",
    "J": 0.075,
}

_motor = estimate_motor_params_from_nameplate(NAMEPLATE_AO2_32_4_3KW)
_omega_nom = 2.0 * math.pi * float(NAMEPLATE_AO2_32_4_3KW["n_rated"]) / 60.0
_torque_nom = float(NAMEPLATE_AO2_32_4_3KW["P_n"]) / max(_omega_nom, 1e-6)
_omega_sync_rpm = 60.0 * float(NAMEPLATE_AO2_32_4_3KW["f_n"]) / max(int(NAMEPLATE_AO2_32_4_3KW["p"]), 1)
_omega_rated_pu = float(NAMEPLATE_AO2_32_4_3KW["n_rated"]) / max(_omega_sync_rpm, 1e-6)
_id_ref = float(estimate_id_ref_from_nameplate(NAMEPLATE_AO2_32_4_3KW, k_m=0.35))

_sim = replace(
    _base.sim,
    t_end=2.0,
    dt=1e-3,
    save_prefix="backlog_ao2_nameplate_first",
    scenario_name="speed_step",
    load_torque=0.25 * _torque_nom,
)

_foc = replace(
    _base.foc,
    id_ref=_id_ref,
    iq_limit=max(float(NAMEPLATE_AO2_32_4_3KW["I_n"]) * 3.0, 2.0),
)

_inverter = replace(
    _base.inverter,
    r_out=0.06,
    dead_time=2e-6,
    v_drop=1.3,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

# Keep AO2 backlog research physically tied to the nameplate instead of the
# older hand-shaped live AO2 branch.
loss_inv_r = 0.280000
loss_core_k = 0.650000
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

id_ref_lut_path = None

# Rated-speed-first backlog branch: default sweeps should stay below synchronous
# speed unless an experiment explicitly reopens the higher-speed target.
ao2_backlog_omega_rated_pu = _omega_rated_pu
ai_omega_ref_pu_range = (0.3, float(_omega_rated_pu))
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

ai_eval_checkpoint_path = ""
ai_eval_id_ref_alpha = 0.20
ai_eval_delta_id_max = 0.15
ai_eval_id_ref_relative = True
ai_eval_id_ref_allow_positive_delta = True
ai_eval_id_ref_gate_speed_tol_rel = 0.10
ai_eval_id_ref_gate_min_scale = 0.20
ai_eval_id_ref_gate_exponent = 1.0
ai_eval_supervisor_enabled = False

__all__ = ["ENV"]
