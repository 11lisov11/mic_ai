from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step27_pipeline import _hybrid_should_activate_secondary


def test_hybrid_trigger_uses_positive_load_delta_when_enabled() -> None:
    assert _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.20,
        omega_ref=100.0,
        omega=100.0,
        load_delta_threshold=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
    )
    assert not _hybrid_should_activate_secondary(
        prev_load=0.20,
        new_load=0.10,
        omega_ref=100.0,
        omega=100.0,
        load_delta_threshold=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
    )


def test_hybrid_trigger_can_use_absolute_load_delta_when_positive_only_disabled() -> None:
    assert _hybrid_should_activate_secondary(
        prev_load=0.20,
        new_load=0.10,
        omega_ref=100.0,
        omega=100.0,
        load_delta_threshold=0.05,
        positive_only=False,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
    )


def test_hybrid_trigger_uses_speed_error_when_configured() -> None:
    assert _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=90.0,
        load_delta_threshold=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
    assert not _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=97.0,
        load_delta_threshold=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
