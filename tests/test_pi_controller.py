# -*- coding: utf-8 -*-
from motor_phys_ai.controllers.pi import PIController


def test_pi_controller_saturation_and_integrator() -> None:
    ctrl = PIController(kp=1.0, ki=1.0, dt=0.1, iq_limit=1.0)

    out = ctrl.step(omega_ref=2.0, omega=0.0, omega_dot=0.0)
    assert out == 1.0
    assert ctrl.integrator == 0.0

    out = ctrl.step(omega_ref=0.5, omega=0.0, omega_dot=0.0)
    assert out > 0.0
    assert ctrl.integrator > 0.0
