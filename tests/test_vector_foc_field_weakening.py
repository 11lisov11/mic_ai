from __future__ import annotations

from control.vector_foc import apply_field_weakening_id_ref


def test_field_weakening_reduces_id_ref_near_voltage_limit() -> None:
    out = apply_field_weakening_id_ref(
        enabled=True,
        current_id_ref=7.0,
        base_id_ref=7.0,
        v_mag=310.0,
        v_limit=311.0,
        id_min=4.0,
        trigger_ratio=0.98,
        relax_ratio=0.92,
        dec_step=0.25,
        relax_step=0.10,
    )
    assert out == 6.75


def test_field_weakening_relaxes_back_to_base_when_headroom_returns() -> None:
    out = apply_field_weakening_id_ref(
        enabled=True,
        current_id_ref=5.5,
        base_id_ref=7.0,
        v_mag=250.0,
        v_limit=311.0,
        id_min=4.0,
        trigger_ratio=0.98,
        relax_ratio=0.92,
        dec_step=0.25,
        relax_step=0.10,
    )
    assert out == 5.6
