import math

import pytest

from tools.uno_q_protocol import (
    CMD_STRUCT,
    TELEMETRY_STRUCT,
    CURRENT_SCALE,
    OMEGA_SCALE,
    POWER_SCALE,
    VDC_SCALE,
    Command,
    Telemetry,
    crc16_ccitt,
)


def test_struct_sizes() -> None:
    assert TELEMETRY_STRUCT.size == 20
    assert CMD_STRUCT.size == 9


def test_telemetry_roundtrip() -> None:
    telem = Telemetry(
        t_ms=12345,
        omega_meas=12.34,
        omega_ref=15.67,
        i_d=1.23,
        i_q=2.34,
        v_dc=48.7,
        i_rms=3.21,
        p_in=56.78,
        status=0x12,
    )
    payload = telem.pack()
    decoded = Telemetry.unpack(payload)

    assert decoded.t_ms == telem.t_ms
    assert decoded.status == telem.status
    assert math.isclose(decoded.omega_meas, telem.omega_meas, abs_tol=1.0 / OMEGA_SCALE)
    assert math.isclose(decoded.omega_ref, telem.omega_ref, abs_tol=1.0 / OMEGA_SCALE)
    assert math.isclose(decoded.i_d, telem.i_d, abs_tol=1.0 / CURRENT_SCALE)
    assert math.isclose(decoded.i_q, telem.i_q, abs_tol=1.0 / CURRENT_SCALE)
    assert math.isclose(decoded.v_dc, telem.v_dc, abs_tol=1.0 / VDC_SCALE)
    assert math.isclose(decoded.i_rms, telem.i_rms, abs_tol=1.0 / CURRENT_SCALE)
    assert math.isclose(decoded.p_in, telem.p_in, abs_tol=1.0 / POWER_SCALE)


def test_air56_nominal_speed_fits_omega_protocol_range() -> None:
    assert OMEGA_SCALE == 128.0
    nominal_rad_s = 2.0 * math.pi * 1380.0 / 60.0
    telem = Telemetry(
        t_ms=1,
        omega_meas=nominal_rad_s,
        omega_ref=nominal_rad_s,
        i_d=1.35,
        i_q=0.7,
        v_dc=24.0,
        i_rms=1.5,
        p_in=250.0,
        status=0,
    )

    decoded = Telemetry.unpack(telem.pack())
    assert math.isclose(decoded.omega_meas, nominal_rad_s, abs_tol=1.0 / OMEGA_SCALE)


def test_telemetry_pack_rejects_out_of_range_speed() -> None:
    telem = Telemetry(
        t_ms=1,
        omega_meas=300.0,
        omega_ref=300.0,
        i_d=1.35,
        i_q=0.7,
        v_dc=24.0,
        i_rms=1.5,
        p_in=250.0,
        status=0,
    )
    with pytest.raises(ValueError, match="omega_meas out of int16 protocol range"):
        telem.pack()


def test_command_crc_roundtrip() -> None:
    cmd = Command(t_ms=54321, enable_ai=1, id_ref=0.42)
    payload = cmd.pack_with_crc()
    assert len(payload) == CMD_STRUCT.size

    crc = int.from_bytes(payload[-2:], byteorder="little")
    calc = crc16_ccitt(payload[:-2] + b"\x00\x00")
    assert crc == calc

    decoded = Command.unpack(payload)
    assert decoded.t_ms == cmd.t_ms
    assert decoded.enable_ai == cmd.enable_ai
    assert math.isclose(decoded.id_ref, cmd.id_ref, abs_tol=1.0 / CURRENT_SCALE)
