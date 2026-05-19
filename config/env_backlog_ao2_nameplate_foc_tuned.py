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
_id_ref_nameplate = float(estimate_id_ref_from_nameplate(NAMEPLATE_AO2_32_4_3KW, k_m=0.35))

_sim = replace(
    _base.sim,
    t_end=2.0,
    dt=1e-3,
    save_prefix="backlog_ao2_nameplate_foc_tuned",
    scenario_name=f"speed_step:{_omega_rated_pu}",
    load_torque=0.25 * _torque_nom,
)

_foc = replace(
    _base.foc,
    # Tuned on 2026-04-12 by cheap no-training probes on the calibrated AO2 branch.
    kp_id=2.0,
    ki_id=200.0,
    kp_iq=2.0,
    ki_iq=200.0,
    kp_speed=0.1,
    ki_speed=1.0,
    # Higher flux with lower torque-current ceiling reduces slip enough to revive AO2
    # on the calibrated quarter-nominal branch.
    id_ref=7.0,
    iq_limit=10.0,
    field_weakening_enable=True,
    field_weakening_id_min=4.5,
    field_weakening_trigger_ratio=0.98,
    field_weakening_relax_ratio=0.92,
    field_weakening_dec_step=0.05,
    field_weakening_relax_step=0.03,
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

ao2_backlog_omega_rated_pu = _omega_rated_pu
ao2_backlog_id_ref_nameplate = _id_ref_nameplate
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

ai_eval_checkpoint_path = "outputs/ao2_tuned_rampfocus_pilot_20260412m/shared/checkpoints/env_backlog_ao2_nameplate_foc_tuned/best_actor.pth"
ai_eval_id_ref_alpha = 0.1646803062368563
ai_eval_delta_id_max = 0.0510458098699188
ai_eval_id_ref_relative = True
ai_eval_id_ref_allow_positive_delta = True
ai_eval_id_ref_gate_speed_tol_rel = 0.1542552317605739
ai_eval_id_ref_gate_min_scale = 0.1770305210531581
ai_eval_id_ref_gate_exponent = 0.9153403067965478
ai_eval_supervisor_enabled = True
ai_eval_sup_objective = "specific_power"
ai_eval_sup_speed_tol_rel = 0.0801045446941467
ai_eval_sup_speed_tol_abs = 0.0
ai_eval_sup_omega_min = 0.1
ai_eval_sup_update = 16
ai_eval_sup_dither = 0.0195168809984451
ai_eval_sup_step = 0.0100396023108874
ai_eval_sup_bias_max = 0.170596010913614
ai_eval_sup_shaft_eps = 10.0
ai_eval_sup_reset_decay = 0.98
ai_eval_sup_idle_enable = True
ai_eval_sup_idle_omega_min = 0.0770502929850793
ai_eval_sup_idle_action = -0.8242774658851884
ai_eval_sup_idle_blend = 0.1
ai_eval_sup_idle_exit_boost = 6
ai_eval_sup_idle_exit_action = 0.9905939208476224
ai_eval_sup_idle_bias_decay = 0.9493637394994622

__all__ = ["ENV"]
