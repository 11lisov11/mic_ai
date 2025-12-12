from __future__ import annotations

from dataclasses import replace

from config.env import create_default_env, estimate_motor_params_from_nameplate


"""
Конфиг для двигателя АИР56 ~0.25 кВт (сеть 380В, 50Гц).

Важно: точные паспортные данные могут отличаться по исполнению/полюсности.
Этот конфиг даёт разумную стартовую модель + параметры сценариев/идентификации/обучения.
"""

_base = create_default_env()

NAMEPLATE_AIR56_025KW = {
    "P_n": 0.25 * 1000.0,  # W
    "U_ll": 380.0,         # V
    "I_n": 0.7,            # A (типично для 0.25кВт на 380В)
    "cos_phi_n": 0.68,
    "eta_n": 0.75,
    "f_n": 50.0,           # Hz
    "p": 2,                # pole pairs (4-pole)
    "n_rated": 1450.0,     # rpm
    "connection": "Y",
    "J": 0.01,
}

_motor_est = estimate_motor_params_from_nameplate(NAMEPLATE_AIR56_025KW)
_motor = replace(
    _motor_est,
    # Сигма-индуктивности в estimate_motor_params_from_nameplate заданы грубо; оставим,
    # дальше их уточнит идентификация.
    Ls_sigma=float(getattr(_motor_est, "Ls_sigma", 0.05)),
    Lr_sigma=float(getattr(_motor_est, "Lr_sigma", 0.05)),
    J=float(NAMEPLATE_AIR56_025KW["J"]),
)

_sim = replace(
    _base.sim,
    t_end=2.0,
    dt=1e-3,
    save_prefix="air56_025kw",
    scenario_name="speed_step",
    load_torque=0.4,
)

ENV = replace(_base, motor=_motor, sim=_sim)

# Параметры тестов идентификации (могут быть подстроены под реальный стенд)
ident_u_d_step = 180.0
ident_total_time = 2.0
ident_u_q_step = 260.0
ident_locked_total_time = 2.5
ident_torque_ref = 2.0
ident_runup_time = 0.8
ident_coast_time = 0.8

# AI defaults
ai_delta_iq_max = 0.8
ai_sigma_omega = 0.05
ai_sigma_id = 0.03
ai_sigma_iq = 0.03
ai_drift_every_episodes = 5
ai_drift_scale = 0.04
ai_w_ext_scale = 1.0
ai_w_int_scale = 0.0
ai_wm_lr = 1e-4
ai_curiosity_beta = 0.0
ai_v_max = 1.0

__all__ = ["ENV"]

