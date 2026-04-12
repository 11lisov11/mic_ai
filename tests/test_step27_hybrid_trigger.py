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
        omega_ref_pu=0.5,
        omega_pu=0.45,
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
        omega_ref_pu=0.5,
        omega_pu=0.485,
        load_delta_threshold=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )


def test_hybrid_speed_trigger_can_be_limited_to_low_speed_window() -> None:
    assert _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=90.0,
        omega_ref_pu=0.20,
        omega_pu=0.18,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
        speed_trigger_max_omega_ref_pu=0.30,
        speed_trigger_max_omega_pu=0.25,
    )
    assert not _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=90.0,
        omega_ref_pu=0.60,
        omega_pu=0.54,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
        speed_trigger_max_omega_ref_pu=0.30,
        speed_trigger_max_omega_pu=0.25,
    )


def test_hybrid_trigger_can_use_voltage_ratio_at_high_speed() -> None:
    assert _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=95.0,
        omega_ref_pu=0.96,
        omega_pu=0.91,
        voltage_ratio=0.995,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
        voltage_trigger_threshold=0.99,
        voltage_trigger_min_omega_ref_pu=0.90,
        voltage_trigger_min_omega_pu=0.85,
    )


def test_hybrid_voltage_trigger_respects_speed_window() -> None:
    assert not _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=95.0,
        omega_ref_pu=0.25,
        omega_pu=0.20,
        voltage_ratio=0.999,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
        voltage_trigger_threshold=0.99,
        voltage_trigger_min_omega_ref_pu=0.90,
        voltage_trigger_min_omega_pu=0.85,
    )


def test_hybrid_voltage_trigger_can_require_speed_error() -> None:
    assert not _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=98.5,
        omega_ref_pu=0.96,
        omega_pu=0.95,
        voltage_ratio=1.01,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
        voltage_trigger_threshold=0.99,
        voltage_trigger_min_omega_ref_pu=0.90,
        voltage_trigger_min_omega_pu=0.85,
        voltage_trigger_require_speed=True,
    )
    assert _hybrid_should_activate_secondary(
        prev_load=0.10,
        new_load=0.10,
        omega_ref=100.0,
        omega=90.0,
        omega_ref_pu=0.96,
        omega_pu=0.86,
        voltage_ratio=1.01,
        load_delta_threshold=0.0,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
        voltage_trigger_threshold=0.99,
        voltage_trigger_min_omega_ref_pu=0.90,
        voltage_trigger_min_omega_pu=0.85,
        voltage_trigger_require_speed=True,
    )
