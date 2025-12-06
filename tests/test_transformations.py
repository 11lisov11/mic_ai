import math

import numpy as np
import pytest

from config.env import ENV, FocParams, InverterParams, MotorParams, ScalarVfParams
from control.scalar_vf import ScalarVfController
from control.vector_foc import FocController
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import (
    abc_to_alpha_beta,
    abc_to_dq,
    alpha_beta_to_abc,
    dq_to_abc,
    dq_to_alpha_beta,
)


def test_transform_round_trip():
    i_abc = (2.0, -1.0, -1.0)
    theta = 0.7
    i_d, i_q = abc_to_dq(*i_abc, theta)
    v_abc = dq_to_abc(i_d, i_q, theta)
    assert np.allclose(i_abc, v_abc, atol=1e-6)


def test_clarke_park_inverse_consistency():
    i_alpha, i_beta = abc_to_alpha_beta(1.0, -0.5, -0.5)
    i_a, i_b, i_c = alpha_beta_to_abc(i_alpha, i_beta)
    assert pytest.approx(1.0, rel=1e-6) == i_a
    assert pytest.approx(-0.5, rel=1e-6) == i_b
    assert pytest.approx(-0.5, rel=1e-6) == i_c


def test_inverter_limits_voltage():
    inverter = IdealInverter(InverterParams(Vdc=100.0, f_pwm=10_000.0))
    v_abc, (v_d, v_q) = inverter.output(100.0, 100.0, theta_e=0.0)
    vmax = inverter.params.Vdc / math.sqrt(3.0)
    assert math.hypot(v_d, v_q) <= vmax + 1e-9
    assert len(v_abc) == 3


def test_scalar_vf_respects_limits():
    params = ScalarVfParams(k_vf=1.0, u_boost=2.0, f_min=5.0, f_max=50.0)
    ctrl = ScalarVfController(params, dt=0.001, p=2, vdc=300.0)
    
    # New signature: t, omega_ref, omega_m, i_abc, torque_e, theta_mech
    v_d, v_q, _, omega_e, info = ctrl.step(
        t=0.0, 
        omega_ref=10.0, 
        omega_m=0.0, 
        i_abc=(0.0, 0.0, 0.0), 
        torque_e=0.0, 
        theta_mech=0.0
    )
    
    f_e = info["f_e"]
    te_filt = info["te_filt"]

    # ramp then min freq clamp to f_min
    assert f_e == pytest.approx(params.f_min)
    assert omega_e == pytest.approx(2 * math.pi * params.f_min, rel=1e-3)
    assert v_d == pytest.approx(0.0)
    assert v_q > 0.0
    assert abs(v_q) == pytest.approx(params.u_boost + params.k_vf * params.f_min)
    assert te_filt == pytest.approx(0.0)


def test_foc_zero_error_outputs_zero_voltage():
    foc = FocController(
        FocParams(
            kp_id=10.0,
            ki_id=0.0,
            kp_iq=10.0,
            ki_iq=0.0,
            kp_speed=1.0,
            ki_speed=0.0,
            iq_limit=10.0,
            v_limit=50.0,
        ),
        MotorParams(**{**ENV.motor.__dict__}),
        dt=1e-4,
    )
    
    v_d, v_q, _, _, info = foc.step(
        t=0.0,
        omega_ref=0.0, 
        omega_m=0.0, 
        i_abc=(0.0, 0.0, 0.0), 
        torque_e=0.0, 
        theta_mech=0.0
    )
    
    i_d_ref = info["i_d_ref"]
    i_q_ref = info["i_q_ref"]

    assert v_d == pytest.approx(0.0)
    assert v_q == pytest.approx(0.0)
    assert i_d_ref == pytest.approx(0.0)
    assert i_q_ref == pytest.approx(0.0)


def test_motor_stays_zero_without_excitation():
    motor = InductionMotorModel(ENV.motor)
    state, i_d, i_q, torque_e, omega_m = motor.step(
        v_ds=0.0, v_qs=0.0, load_torque=0.0, dt=1e-3, omega_syn=0.0
    )
    assert i_d == pytest.approx(0.0)
    assert i_q == pytest.approx(0.0)
    assert torque_e == pytest.approx(0.0)
    assert omega_m == pytest.approx(0.0)
    assert state.psi_ds == pytest.approx(0.0)
    assert state.psi_qs == pytest.approx(0.0)
