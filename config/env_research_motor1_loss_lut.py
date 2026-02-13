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
    save_prefix="research_motor1_loss_lut",
    scenario_name="speed_step",
    load_torque=0.4,
)

_foc = replace(
    _base.foc,
    id_ref=1.4,
    iq_limit=2.0,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

loss_inv_r = 3.706943
loss_core_k = 0.090869
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

# Adaptive MIC policy distilled to LUT (bounded safe grid).
id_ref_lut_path = "outputs/research20260212/id_ref_lut_research_safe/id_ref_lut.json"

__all__ = ["ENV"]
