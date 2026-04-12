from __future__ import annotations

from tools.air56_unoq_bridge import (
    Air56IdRefParams,
    _action_to_id_ref,
    _estimate_load_nm_from_iq,
    _should_switch_secondary,
)


def _params() -> Air56IdRefParams:
    return Air56IdRefParams(
        id_ref_base=1.35,
        id_ref_min=1.10,
        id_ref_max=1.70,
        id_ref_alpha=0.2,
        delta_id_max=0.1,
        gate_speed_tol_abs=0.0,
        gate_speed_tol_rel=0.1,
        gate_min_scale=0.1,
        gate_exponent=1.0,
        ai_id_relative=True,
        allow_positive_delta=True,
    )


def test_estimate_load_nm_from_iq_uses_absolute_current() -> None:
    assert _estimate_load_nm_from_iq(-0.8, 2.5) == 2.0


def test_action_to_id_ref_blocks_demagnetization_on_large_error() -> None:
    cmd, gate_scale, _gate_tol = _action_to_id_ref(
        action=-1.0,
        prev_id_ref=1.35,
        omega_ref=100.0,
        omega_meas=0.0,
        params=_params(),
    )
    assert gate_scale <= 0.1
    assert cmd >= 1.35


def test_should_switch_secondary_requires_real_positive_load_jump() -> None:
    assert _should_switch_secondary(
        load_est_nm=0.60,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
    assert not _should_switch_secondary(
        load_est_nm=0.42,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
