from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env, InverterParams


_base = create_default_env()

_motor = replace(
    _base.motor,
    Rs=3.2,
    Rr=2.8,
    Ls_sigma=0.08,
    Lr_sigma=0.08,
    Lm=0.25,
    J=0.012,
    B=2e-3,
    # Saturation model (approximate)
    psi_sat=0.12,
    sat_exp=2.0,
    lm_min_scale=0.3,
)

_inverter = replace(
    _base.inverter,
    r_out=0.12,
    dead_time=2e-6,
    v_drop=1.2,
)

_sim = replace(
    _base.sim,
    t_end=2.0,
    dt=1e-3,
    save_prefix="demo_motor1_physical",
    scenario_name="speed_step",
    load_torque=0.4,
)

_foc = replace(
    _base.foc,
    kp_speed=2.0,
    ki_speed=20.0,
    id_ref=0.36,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

# Loss model (optional, used in p_in_total)
loss_inv_r = 3.706943
loss_core_k = 0.090869
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

ident_u_d_step = 200.0
ident_total_time = 2.0
ident_u_q_step = 260.0
ident_locked_total_time = 2.5
ident_torque_ref = 2.0
ident_runup_time = 0.8
ident_coast_time = 0.8

# AI assist defaults
ai_delta_iq_max = 0.8
ai_sigma_omega = 0.05
ai_sigma_id = 0.03
ai_sigma_iq = 0.03
ai_drift_every_episodes = 1
ai_drift_scale = 0.04
ai_w_ext_scale = 1.0
ai_w_int_scale = 0.5
ai_wm_lr = 0.0001
ai_curiosity_beta = 1.0
ai_w_id_current = 0.2
baseline_speed_err = 2.99123
baseline_current_rms = 1.34218
ext_scale = 4.333
ai_v_max = 1.0

# Optional LUT for id_ref scheduling (omega_ref, load_torque -> id_ref)
id_ref_lut_path = "outputs/id_ref_lut_motor1/id_ref_lut.json"

# Domain randomization defaults for generalization.
ai_omega_ref_pu_range = (0.3, 1.1)
ai_load_mult_range = (0.5, 1.5)
ai_drift_params = ("Rs", "Rr", "Lm", "Ls_sigma", "Lr_sigma", "J", "B")
ai_drift_ranges = {
    "Rs": (0.7, 1.3),
    "Rr": (0.7, 1.3),
    "Lm": (0.8, 1.2),
    "Ls_sigma": (0.7, 1.3),
    "Lr_sigma": (0.7, 1.3),
    "J": (0.5, 1.5),
    "B": (0.5, 2.0),
}

__all__ = ["ENV"]
