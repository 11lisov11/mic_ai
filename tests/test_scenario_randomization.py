from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.scenario_randomization import wrap_scenario_with_ranges


def test_wrap_scenario_with_ranges_scales_profile_peaks() -> None:
    def omega_ref(t: float) -> float:
        return 0.0 if t < 0.5 else 10.0

    def load_torque(t: float) -> float:
        return 1.0 if t < 0.5 else 2.0

    rng = np.random.default_rng(123)
    wrapped_omega, wrapped_load, meta = wrap_scenario_with_ranges(
        omega_ref,
        load_torque,
        t_end=1.0,
        rng=rng,
        omega_ref_range=(20.0, 20.0),
        load_torque_range=(3.0, 3.0),
    )

    assert abs(meta["omega_base_peak"] - 10.0) < 1e-9
    assert abs(meta["load_base_peak"] - 2.0) < 1e-9
    assert abs(meta["omega_scale"] - 2.0) < 1e-9
    assert abs(meta["load_scale"] - 1.5) < 1e-9
    assert abs(meta["omega_peak"] - 20.0) < 1e-9
    assert abs(meta["load_peak"] - 3.0) < 1e-9
    assert abs(wrapped_omega(0.25) - 0.0) < 1e-9
    assert abs(wrapped_omega(0.75) - 20.0) < 1e-9
    assert abs(wrapped_load(0.25) - 1.5) < 1e-9
    assert abs(wrapped_load(0.75) - 3.0) < 1e-9


def test_wrap_scenario_with_ranges_no_ranges_keeps_profile() -> None:
    def omega_ref(t: float) -> float:
        return t

    def load_torque(t: float) -> float:
        return 2.0 * t

    rng = np.random.default_rng(123)
    wrapped_omega, wrapped_load, meta = wrap_scenario_with_ranges(
        omega_ref,
        load_torque,
        t_end=1.0,
        rng=rng,
        omega_ref_range=None,
        load_torque_range=None,
    )

    assert abs(meta["omega_scale"] - 1.0) < 1e-9
    assert abs(meta["load_scale"] - 1.0) < 1e-9
    assert abs(meta["omega_peak"] - 1.0) < 1e-9
    assert abs(meta["load_peak"] - 2.0) < 1e-9
    assert abs(wrapped_omega(0.4) - 0.4) < 1e-9
    assert abs(wrapped_load(0.4) - 0.8) < 1e-9
