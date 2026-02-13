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

__all__ = ["ENV"]
