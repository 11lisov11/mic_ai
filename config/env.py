"""
Конфигурация проекта MIC AI.

Все настраиваемые параметры собраны сверху с русскими комментариями.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# -------- Параметры симуляции ----------
SIM_T_END = 3.0            # время моделирования, с
SIM_DT = 1e-4              # дискретизация, с
SIM_MODE = "foc"           # "scalar" или "foc"
SIM_SCENARIO = "speed_step"  # сценарий: speed_step / ramp и т.д.
SIM_SAVE_PREFIX = "run"    # префикс для файла результатов
SIM_LOAD_TORQUE = 1.0      # постоянный момент нагрузки, Нм
# ---------------------------------------

# ------ Паспортные данные электродвигателя -------
NAMEPLATE_P_KW = 0.25       # номинальная мощность, кВт
NAMEPLATE_U_LL = 380.0      # линейное напряжение, В
NAMEPLATE_I_N = 0.7         # линейный ток, А
NAMEPLATE_COSPHI = 0.68     # cos φ номинальный
NAMEPLATE_ETA = 0.75        # КПД (0..1)
NAMEPLATE_F_N = 50.0        # частота сети, Гц
NAMEPLATE_POLE_PAIRS = 2    # число пар полюсов
NAMEPLATE_N_RATED = 1450.0  # номинальная скорость, об/мин
NAMEPLATE_CONNECTION = "Y"  # соединение "Y" или "D"
NAMEPLATE_J = 0.01          # инерция, кг*м^2 (оценка)
# -------------------------------------------------

# ------ Параметры инвертора -------
INVERTER_VDC = 540.0       # звено постоянного тока, В
INVERTER_F_PWM = 10_000.0  # частота ШИМ, Гц
# ----------------------------------

NAMEPLATE_DEFAULT = {
    "P_n": NAMEPLATE_P_KW * 1000.0,
    "U_ll": NAMEPLATE_U_LL,
    "I_n": NAMEPLATE_I_N,
    "cos_phi_n": NAMEPLATE_COSPHI,
    "eta_n": NAMEPLATE_ETA,
    "f_n": NAMEPLATE_F_N,
    "p": NAMEPLATE_POLE_PAIRS,
    "n_rated": NAMEPLATE_N_RATED,
    "connection": NAMEPLATE_CONNECTION,
    "J": NAMEPLATE_J,
}

VF_K = NAMEPLATE_U_LL / (math.sqrt(3.0) * NAMEPLATE_F_N)


# --------- Структуры данных ------------
@dataclass(frozen=True)
class MotorParams:
    Rs: float
    Rr: float
    Ls_sigma: float
    Lr_sigma: float
    Lm: float
    J: float
    B: float
    p: int


@dataclass(frozen=True)
class InverterParams:
    Vdc: float
    f_pwm: float


@dataclass(frozen=True)
class ScalarVfParams:
    k_vf: float
    u_boost: float
    f_min: float
    f_max: float


@dataclass(frozen=True)
class FocParams:
    kp_id: float
    ki_id: float
    kp_iq: float
    ki_iq: float
    kp_speed: float
    ki_speed: float
    id_ref: float = 0.0
    iq_limit: float | None = None
    v_limit: float | None = None


@dataclass(frozen=True)
class SimulationParams:
    t_end: float
    dt: float
    mode: str
    scenario_name: str
    save_prefix: str
    load_torque: float = 0.0


@dataclass(frozen=True)
class EnvConfig:
    motor: MotorParams
    inverter: InverterParams
    scalar_vf: ScalarVfParams
    foc: FocParams
    sim: SimulationParams
# ---------------------------------------


def estimate_id_ref_from_nameplate(nameplate: dict, k_m: float = 0.35) -> float:
    """
    Оценка опорного магнитообразующего тока d-оси по паспортным данным.
    """
    connection = str(nameplate.get("connection", "Y")).upper()
    I_n = float(nameplate.get("I_n", 3.0))
    if connection == "D":
        I_ph = I_n / math.sqrt(3.0)
    else:
        I_ph = I_n
    return k_m * I_ph


def estimate_motor_params_from_nameplate(nameplate: dict) -> MotorParams:
    """
    Приближённый расчёт MotorParams по паспортным данным.
    """
    default_eta = 0.9
    default_cosphi = 0.85
    default_f = 50.0
    default_p = 2
    default_J = 0.05
    default_B = 1e-3
    default_slip = 0.03

    P_n = float(nameplate.get("P_n", 1000.0))
    U_ll = float(nameplate.get("U_ll", 400.0))
    I_n = float(nameplate.get("I_n", 3.0))
    cos_phi_n = float(nameplate.get("cos_phi_n", default_cosphi))
    eta_n = float(nameplate.get("eta_n", default_eta))
    f_n = float(nameplate.get("f_n", default_f))
    p = int(nameplate.get("p", default_p))
    connection = str(nameplate.get("connection", "Y")).upper()

    # Фазные величины
    if connection == "D":
        U_ph = U_ll
        I_ph = I_n / math.sqrt(3.0)
    else:
        U_ph = U_ll / math.sqrt(3.0)
        I_ph = I_n

    # Мощности
    P_out = P_n
    P_in = P_out / eta_n if eta_n > 0 else P_out
    P_loss_total = max(P_in - P_out, 0.0)

    # Скольжение и скорость
    n_sync = 60.0 * f_n / p if p else 0.0
    n_rated = float(nameplate.get("n_rated", n_sync * (1 - default_slip)))
    s_n = (n_sync - n_rated) / n_sync if n_sync > 0 else default_slip
    s_n = max(min(s_n, 0.2), 0.0)
    omega_rated = 2.0 * math.pi * n_rated / 60.0

    # Потери
    # более простое и управляемое разбиение потерь
    P_loss_total = max(P_loss_total, 0.0)
    P_cu_r = 0.4 * P_loss_total
    P_cu_s = 0.3 * P_loss_total
    P_mech_fe = P_loss_total - P_cu_r - P_cu_s

    # Сопротивления
    Rs = P_cu_s / (3.0 * I_ph**2) if I_ph > 0 else 0.5
    phi_n = math.acos(max(min(cos_phi_n, 1.0), -1.0))
    I_2 = I_ph * math.sin(phi_n)
    Rr = P_cu_r / (3.0 * I_2**2) if I_2 > 0 else Rs

    # Индуктивности
    k_m = 0.35
    Im = k_m * I_ph
    Xm = U_ph / Im if Im > 0 else 0.0
    Lm = Xm / (2.0 * math.pi * f_n) if f_n > 0 else 1e-3

    # Рассеяние фиксируем для снижения пульсаций
    Ls_sigma = 0.05
    Lr_sigma = 0.05

    # Механика
    B = P_mech_fe / (omega_rated**2) if omega_rated > 0 else default_B
    J = float(nameplate.get("J", default_J))

    return MotorParams(
        Rs=Rs,
        Rr=Rr,
        Ls_sigma=Ls_sigma,
        Lr_sigma=Lr_sigma,
        Lm=Lm,
        J=J,
        B=B,
        p=p,
    )


# --------- Готовая конфигурация ENV ------------
NAMEPLATE_ID_REF = estimate_id_ref_from_nameplate(NAMEPLATE_DEFAULT)

def create_default_env() -> EnvConfig:
    return EnvConfig(
        motor=estimate_motor_params_from_nameplate(NAMEPLATE_DEFAULT),
        inverter=InverterParams(
            Vdc=INVERTER_VDC,
            f_pwm=INVERTER_F_PWM,
        ),
        scalar_vf=ScalarVfParams(
            k_vf=VF_K,
            u_boost=25.0,
            f_min=1.0,
            f_max=50.0,
        ),
        foc=FocParams(
            kp_id=1.0,
            ki_id=100.0,
            kp_iq=1.0,
            ki_iq=100.0,
            kp_speed=0.5,
            ki_speed=2.5,
            id_ref=0.4,
            iq_limit=2.0,
            v_limit=INVERTER_VDC / math.sqrt(3.0),
        ),
        sim=SimulationParams(
            t_end=SIM_T_END,
            dt=SIM_DT,
            mode=SIM_MODE,
            scenario_name=SIM_SCENARIO,
            save_prefix=SIM_SAVE_PREFIX,
            load_torque=SIM_LOAD_TORQUE,
        ),
    )

ENV = create_default_env()
# -----------------------------------------------


__all__ = [
    "MotorParams",
    "InverterParams",
    "ScalarVfParams",
    "FocParams",
    "SimulationParams",
    "EnvConfig",
    "ENV",
    "estimate_motor_params_from_nameplate",
    "estimate_id_ref_from_nameplate",
    "NAMEPLATE_DEFAULT",
    "SIM_T_END",
    "SIM_DT",
    "SIM_MODE",
    "SIM_SCENARIO",
    "SIM_SAVE_PREFIX",
    "SIM_LOAD_TORQUE",
    "INVERTER_VDC",
    "INVERTER_F_PWM",
    "NAMEPLATE_ID_REF",
    "create_default_env", 
]
