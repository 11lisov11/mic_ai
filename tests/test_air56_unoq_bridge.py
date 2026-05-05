from __future__ import annotations

from tools.air56_unoq_bridge import (
    Air56IdRefParams,
    _action_to_id_ref,
    _estimate_load_nm_from_iq,
    _resolve_existing_file,
    _send_fallback_command,
    _should_switch_secondary,
)
from tools.uno_q_protocol import Command


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


def test_resolve_existing_file_accepts_relative_repo_file() -> None:
    path = _resolve_existing_file("config/env_research_air56_025kw.py", "--config")
    assert path.is_file()
    assert path.name == "env_research_air56_025kw.py"


def test_send_fallback_command_disables_ai() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.payloads.append(payload)

    transport = FakeTransport()
    _send_fallback_command(transport, t_ms=10, id_ref_base=1.35, crc=True)  # type: ignore[arg-type]
    assert len(transport.payloads) == 1

    cmd = Command.unpack(transport.payloads[0])
    assert cmd.t_ms == 10
    assert cmd.enable_ai == 0
    assert abs(cmd.id_ref - 1.35) < 1.0 / 1024.0
