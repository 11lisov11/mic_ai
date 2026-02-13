from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env


_base = create_default_env()

_motor = replace(
    _base.motor,
    Rs=1.8,
    Rr=1.6,
    Ls_sigma=0.06,
    Lr_sigma=0.06,
    Lm=0.18,
    J=0.02,
    B=1.5e-3,
)
_sim = replace(
    _base.sim,
    t_end=2.5,
    dt=1.2e-3,
    save_prefix="demo_motor2",
    scenario_name="speed_step",
    load_torque=0.6,
)
ENV = replace(_base, motor=_motor, sim=_sim)

ident_u_d_step = 170.0
ident_total_time = 2.2
ident_u_q_step = 240.0
ident_locked_total_time = 2.6
ident_torque_ref = 2.8
ident_runup_time = 1.0
ident_coast_time = 1.0

# AI assist defaults
ai_delta_iq_max = 0.8
ai_sigma_omega = 0.05
ai_sigma_id = 0.03
ai_sigma_iq = 0.03
ai_drift_scale = 0.04
ai_w_ext_scale = 1.0
ai_w_int_scale = 0.5
ai_wm_lr = 0.0001
ai_curiosity_beta = 1.0
baseline_speed_err = 2.68643
baseline_current_rms = 2.46702
ext_scale = 5.153
ai_v_max = 1.0

# Domain randomization defaults for generalization.
ai_omega_ref_pu_range = (0.3, 1.1)
ai_load_mult_range = (0.5, 1.5)
ai_drift_every_episodes = 1
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
