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

__all__ = ["ENV"]
