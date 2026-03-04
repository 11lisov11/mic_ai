from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate


_base = create_default_env()

NAMEPLATE_AO2_32_4_3KW = {
    "P_n": 3.00 * 1000.0,  # W
    "U_ll": 380.0,  # V
    "I_n": 7.20,  # A
    "cos_phi_n": 0.82,
    "eta_n": 0.84,
    "f_n": 50.0,  # Hz
    "p": 2,  # pole pairs (4-pole)
    "n_rated": 1430.0,  # rpm
    "connection": "Y",
    "J": 0.075,
}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_AO2_32_4_3KW)
_motor = replace(
    _motor_est,
    # Electrical dynamics tuned into a stable region for long scenario rollouts.
    Rs=1.80,
    Rr=1.60,
    Lm=0.18,
    Ls_sigma=0.06,
    Lr_sigma=0.06,
    J=0.020,
    B=1.5e-3,
)

_load_torque = 1.0

_sim = replace(
    _base.sim,
    t_end=2.0,
    # NOTE: This motor model + default PI gains are tuned around dt=1e-3.
    # Smaller dt significantly degrades speed tracking in our current setup.
    dt=1e-3,
    save_prefix="research_ao2_32_4_3kw",
    scenario_name="speed_step",
    load_torque=float(_load_torque),
)

_foc = replace(
    _base.foc,
    id_ref=1.80,
    iq_limit=12.0,
)

_inverter = replace(
    _base.inverter,
    r_out=0.06,
    dead_time=2e-6,
    v_drop=1.3,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

loss_inv_r = 0.280000
loss_core_k = 0.650000
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
ai_eval_checkpoint_path = "outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth"
ai_eval_id_ref_alpha = 0.1376345145348217
ai_eval_delta_id_max = 0.1142035777048785
ai_eval_id_ref_relative = True
ai_eval_id_ref_allow_positive_delta = True
ai_eval_id_ref_gate_speed_tol_rel = 0.0834656357930326
ai_eval_id_ref_gate_min_scale = 0.1123472010231054
ai_eval_id_ref_gate_exponent = 1.0654488757325706

# AO2 hardening v2 profile (2026-03-04):
# source: outputs/ao2_hardening_v2_20260304_localsafe/ao2_selected_candidate_v2.json
ai_eval_supervisor_enabled = True
ai_eval_sup_objective = "p_in"
ai_eval_sup_speed_tol_rel = 0.081034097630596
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.1
ai_eval_sup_update = 18
ai_eval_sup_dither = 0.0170408811738441
ai_eval_sup_step = 0.0106539671267185
ai_eval_sup_bias_max = 0.1826057593079168
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.98
ai_eval_sup_idle_enable = False
ai_eval_sup_idle_omega_min = 0.049340256028464
ai_eval_sup_idle_action = -0.923947842717912
ai_eval_sup_idle_exit_boost = 6
ai_eval_sup_idle_exit_action = 0.9768517402041176
ai_eval_sup_idle_bias_decay = 0.9524250958707856

__all__ = ["ENV"]
