from __future__ import annotations

import math

import numpy as np

from mic_ai.analysis.metrics import calc_cos_phi, calc_eta, calc_i_rms, calc_p_el, calc_p_mech, calc_v_rms


def _balanced_phase_signals(phi_rad: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 4.0 * math.pi, n, endpoint=False)
    va = np.sin(t)
    vb = np.sin(t - 2.0 * math.pi / 3.0)
    vc = np.sin(t + 2.0 * math.pi / 3.0)
    ia = np.sin(t - phi_rad)
    ib = np.sin(t - 2.0 * math.pi / 3.0 - phi_rad)
    ic = np.sin(t + 2.0 * math.pi / 3.0 - phi_rad)
    return np.stack([va, vb, vc], axis=1), np.stack([ia, ib, ic], axis=1)


def test_calc_cos_phi_phase_voltage_model() -> None:
    phi = math.radians(30.0)
    v, i = _balanced_phase_signals(phi)
    cos_phi, diag = calc_cos_phi(v, i)
    assert diag["method"] == "phase"
    assert abs(cos_phi - math.cos(phi)) < 0.02


def test_calc_cos_phi_line_voltage_model_warns_on_disagreement() -> None:
    phi = math.radians(25.0)
    v_phase, i = _balanced_phase_signals(phi)
    v_ll = np.stack(
        [
            v_phase[:, 0] - v_phase[:, 1],
            v_phase[:, 1] - v_phase[:, 2],
            v_phase[:, 2] - v_phase[:, 0],
        ],
        axis=1,
    )
    cos_phi, diag = calc_cos_phi(v_ll, i)
    assert diag["method"] in {"phase", "line"}
    assert str(diag.get("warning", "")) == "phase_line_disagreement"
    assert float(diag["cos_line_raw"]) > float(diag["cos_phase_raw"])
    assert 0.0 <= cos_phi <= 1.0


def test_calc_cos_phi_clip_bounds() -> None:
    v = np.ones((128, 3), dtype=float)
    i = np.ones((128, 3), dtype=float) * 50.0
    cos_phi, _ = calc_cos_phi(v, i)
    assert 0.0 <= cos_phi <= 1.0


def test_power_metrics_and_eta_bounds() -> None:
    v = np.array([230.0, -115.0, -115.0], dtype=float)
    i = np.array([2.0, -1.0, -1.0], dtype=float)
    p_el = calc_p_el(v, i)
    assert abs(p_el - 690.0) < 1e-9

    p_mech = calc_p_mech(100.0, 5.0)
    assert abs(p_mech - 500.0) < 1e-9

    eta = calc_eta(p_mech, p_el)
    assert 0.0 <= eta <= 1.0
    assert abs(eta - (500.0 / 690.0)) < 1e-9


def test_calc_eta_handles_invalid_input() -> None:
    assert calc_eta(100.0, 0.0) == 0.0
    assert calc_eta(float("nan"), 10.0) == 0.0
    assert calc_eta(10.0, float("inf")) == 0.0


def test_rms_and_power_handle_non_finite_input() -> None:
    i = np.array([[np.inf, 2.0, -np.inf], [np.nan, 1.0, 3.0]], dtype=float)
    v = np.array([[400.0, np.nan, -400.0], [np.inf, -200.0, -200.0]], dtype=float)
    i_rms = calc_i_rms(i)
    v_rms = calc_v_rms(v)
    p_el = calc_p_el(v, i)
    assert math.isfinite(i_rms)
    assert math.isfinite(v_rms)
    assert math.isfinite(p_el)
    assert i_rms >= 0.0
    assert v_rms >= 0.0


def test_calc_cos_phi_non_finite_inputs_returns_bounded_value() -> None:
    v = np.array([[np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0]], dtype=float)
    i = np.array([[np.inf, np.nan, -np.inf], [0.0, 0.0, 0.0]], dtype=float)
    cos_phi, diag = calc_cos_phi(v, i)
    assert 0.0 <= cos_phi <= 1.0
    assert str(diag.get("method", "")) in {"phase", "line", "none"}
