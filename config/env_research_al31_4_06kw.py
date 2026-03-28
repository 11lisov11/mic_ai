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

# Step27/Step28 AI evaluation defaults (sensorless MIC).
# Promoted strict pair:
#   checkpoint = outputs/al31_anchor_ep008_medium4_20260328a/results_run/20260328_110425_tmp_al31_mid04_train_20260322_ai_id_ref/eval/actor_ep_init.pth
#   candidate  = outputs/al31_mid04_ultrafine2_20260328l/al31_tuning_summary.json#mid04_speed_dn_04
ai_eval_checkpoint_path = "outputs/al31_anchor_ep008_medium4_20260328a/results_run/20260328_110425_tmp_al31_mid04_train_20260322_ai_id_ref/eval/actor_ep_init.pth"
ai_eval_id_ref_alpha = 0.09675
ai_eval_delta_id_max = 0.1052
ai_eval_id_ref_relative = True
ai_eval_id_ref_allow_positive_delta = True
ai_eval_id_ref_gate_speed_tol_rel = 0.1192
ai_eval_id_ref_gate_min_scale = 0.2038
ai_eval_id_ref_gate_exponent = 0.958

ai_eval_supervisor_enabled = True
ai_eval_sup_objective = 'specific_power'
ai_eval_sup_speed_tol_rel = 0.079
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.1
ai_eval_sup_update = 23
ai_eval_sup_dither = 0.02414
ai_eval_sup_step = 0.00503
ai_eval_sup_bias_max = 0.1635
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.98
ai_eval_sup_idle_enable = False
ai_eval_sup_objective_clip = 3.12

__all__ = ["ENV"]
ai_eval_sup_idle_omega_min = 0.04826439310210644
ai_eval_sup_idle_action = -0.5
ai_eval_sup_idle_exit_boost = 11
ai_eval_sup_idle_exit_action = 0.9723990598816908
ai_eval_sup_idle_bias_decay = 0.96
