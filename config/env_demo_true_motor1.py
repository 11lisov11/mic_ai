from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env


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
)
_sim = replace(
    _base.sim,
    t_end=2.0,
    dt=1e-3,
    save_prefix="demo_motor1",
    scenario_name="speed_step",
    load_torque=0.4,
)
ENV = replace(_base, motor=_motor, sim=_sim)

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
ai_drift_every_episodes = 5
ai_drift_scale = 0.04
ai_w_ext_scale = 1.0
ai_w_int_scale = 0.5
ai_wm_lr = 0.0001
ai_curiosity_beta = 1.0
baseline_speed_err = 2.99123
baseline_current_rms = 1.34218
ext_scale = 4.333
ai_v_max = 1.0

__all__ = ["ENV"]
