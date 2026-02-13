import numpy as np

from mic_ai.tools import scenario_compare


def test_err_limit() -> None:
    limit = scenario_compare._err_limit(1.0, rel_tol=0.1, abs_tol=0.5)
    assert abs(limit - 1.1) < 1e-6
    limit2 = scenario_compare._err_limit(0.1, rel_tol=0.1, abs_tol=0.5)
    assert abs(limit2 - 0.5) < 1e-6


def test_summarize_basic() -> None:
    series = {
        "t": np.linspace(0.0, 1.0, 10),
        "omega": np.ones(10),
        "omega_ref": np.ones(10),
        "i_rms": np.ones(10),
        "p_el": np.ones(10) * 2.0,
        "p_mech": np.ones(10) * 1.0,
    }
    out = scenario_compare._summarize(series, window_frac=0.5)
    assert out["mean_abs_speed_err"] == 0.0
    assert abs(out["eta"] - 0.5) < 1e-6
