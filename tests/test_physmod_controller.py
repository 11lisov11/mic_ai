# -*- coding: utf-8 -*-
from motor_phys_ai.controllers.physmod import PhysModController


def test_physmod_adapts_kt() -> None:
    ctrl = PhysModController(
        kp=1.0,
        ki=0.0,
        dt=0.1,
        inertia_j=1.0,
        kt_init=1.0,
        iq_limit=None,
        kt_alpha=1.0,
        omega_dot_threshold=0.1,
        kt_min=0.01,
        kt_max=10.0,
    )

    # Positive error dominates feed-forward, iq_cmd positive -> kt update should happen.
    _ = ctrl.step(omega_ref=10.0, omega=0.0, omega_dot=1.0)
    assert ctrl.kt_hat < 1.0
    assert ctrl.kt_hat > 0.01
