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
    save_prefix="study_motor1",
    scenario_name="speed_step",
    load_torque=0.2,
)

_foc = replace(
    _base.foc,
    kp_speed=4.0,
    ki_speed=40.0,
    id_ref=0.4,
    iq_limit=6.0,
)

ENV = replace(_base, motor=_motor, sim=_sim, foc=_foc)

# Loss model (used in p_in_total)
loss_inv_r = 3.706943
loss_core_k = 0.090869
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

# Disable LUT to keep a consistent baseline in comparisons.
id_ref_lut_path = None

__all__ = ["ENV"]
