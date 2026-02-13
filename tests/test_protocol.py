from tools.uno_q_protocol import Command, Telemetry, crc16_ccitt


def test_protocol_roundtrip() -> None:
    telemetry = Telemetry(
        t_ms=12345,
        omega_meas=12.5,
        omega_ref=13.0,
        i_d=1.25,
        i_q=-0.75,
        v_dc=220.0,
        i_rms=1.4,
        p_in=55.5,
        status=0xA5,
    )
    payload = telemetry.pack()
    parsed = Telemetry.unpack(payload)
    assert parsed.t_ms == telemetry.t_ms
    assert abs(parsed.omega_meas - telemetry.omega_meas) < 1e-3
    assert abs(parsed.i_q - telemetry.i_q) < 1e-3
    assert parsed.status == telemetry.status

    command = Command(t_ms=999, enable_ai=1, id_ref=0.75, crc=0xBEEF)
    cmd_payload = command.pack()
    cmd_parsed = Command.unpack(cmd_payload)
    assert cmd_parsed.t_ms == command.t_ms
    assert cmd_parsed.enable_ai == command.enable_ai
    assert abs(cmd_parsed.id_ref - command.id_ref) < 1e-3

    zero_payload = Command(t_ms=999, enable_ai=1, id_ref=0.75, crc=0).pack()
    expected_crc = crc16_ccitt(zero_payload)
    crc_payload = Command(t_ms=999, enable_ai=1, id_ref=0.75, crc=0).pack_with_crc()
    parsed_crc = Command.unpack(crc_payload)
    assert parsed_crc.crc == expected_crc
