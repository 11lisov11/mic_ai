import math

from config.env import MotorParams
from models.induction_motor import InductionMotorModel, MotorState


def test_motor_saturation_effect() -> None:
    params = MotorParams(
        Rs=1.0,
        Rr=1.0,
        Ls_sigma=0.05,
        Lr_sigma=0.05,
        Lm=0.3,
        J=0.01,
        B=1e-3,
        p=2,
        psi_sat=0.05,
        sat_exp=2.0,
        lm_min_scale=0.3,
    )
    model = InductionMotorModel(params)
    state = MotorState(psi_ds=0.2, psi_qs=0.0, psi_dr=0.05, psi_qr=0.0, omega_m=0.0)
    lm_eff = model._lm_effective(state)
    assert lm_eff < params.Lm

    next_state, i_d, i_q, torque_e, omega_m = model.step(1.0, 0.0, 0.0, 1e-4)
    assert math.isfinite(i_d)
    assert math.isfinite(torque_e)
