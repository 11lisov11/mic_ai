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
    save_prefix="research_motor1_loss_nominal",
    scenario_name="speed_step",
    load_torque=0.4,
)

# Higher base flux is used here to keep the speed loop in a stable operating
# region; MIC then adapts id_ref down inside steady-state gates.
_foc = replace(
    _base.foc,
    id_ref=1.4,
    iq_limit=2.0,
)

ENV = replace(_base, motor=_motor, inverter=_inverter, sim=_sim, foc=_foc)

# Loss model used by p_in_total.
loss_inv_r = 3.706943
loss_core_k = 0.090869
loss_core_omega_exp = 0.5
loss_core_psi_exp = 0.0

# Disable LUT for strict apples-to-apples FOC vs MIC comparison.
id_ref_lut_path = None

__all__ = ["ENV"]
