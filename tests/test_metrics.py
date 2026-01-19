# -*- coding: utf-8 -*-
import numpy as np

from motor_phys_ai.utils.metrics import calc_metrics


def test_metrics_values() -> None:
    t = np.array([0.0, 1.0, 2.0])
    omega_ref = np.array([1.0, 1.0, 1.0])
    omega = np.array([1.0, 0.0, 2.0])
    i_q = np.array([1.0, 2.0, 3.0])

    metrics = calc_metrics(t, omega, omega_ref, i_q=i_q, iq_limit=2.5, speed_tol_rel=0.5)
    assert abs(metrics["J_speed"] - (2.0 / 3.0)) < 1e-6
    assert abs(metrics["J_energy"] - 14.0) < 1e-6
    assert metrics["J_stability"] > 0.0
