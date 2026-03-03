from __future__ import annotations

import math

import numpy as np

from mic_ai.analysis.metrics import calc_cos_phi


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
