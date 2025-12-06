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

__all__ = ["ENV"]
