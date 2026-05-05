from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "arduino" / "air56_unoq_ready"
FW = PKG / "firmware" / "air56_unoq_example"


def test_air56_unoq_firmware_uses_hardware_adapter() -> None:
    ino = (FW / "air56_unoq_example.ino").read_text(encoding="utf-8")
    assert '#include "air56_unoq_hw.h"' in ino
    assert "return 0.0f" not in ino
    assert "(void)id_ref_amp" not in ino
    assert "air56_hw_apply_id_ref_amp" in ino


def test_air56_unoq_firmware_latches_last_command_until_timeout() -> None:
    ino = (FW / "air56_unoq_example.ino").read_text(encoding="utf-8")
    assert "g_last_cmd_id_ref_q10" in ino
    assert "g_last_cmd_enable_ai" in ino
    assert "if (!timeout) {" in ino
    assert "!timeout && have_cmd" not in ino


def test_air56_unoq_firmware_rejects_invalid_enable_ai_commands() -> None:
    ino = (FW / "air56_unoq_example.ino").read_text(encoding="utf-8")
    assert "if (cmd->enable_ai > 1u)" in ino


def test_air56_unoq_firmware_speed_guard_uses_wide_error_math() -> None:
    ino = (FW / "air56_unoq_example.ino").read_text(encoding="utf-8")
    control = (FW / "uno_q_control.h").read_text(encoding="utf-8")
    root_control = (ROOT / "arduino" / "uno_q_control.h").read_text(encoding="utf-8")

    assert "const int32_t speed_delta_q10" in ino
    assert "const int32_t speed_err_q10" in ino
    assert "const int16_t speed_err_q10" not in ino
    assert "int32_t speed_err_q10" in control
    assert "int32_t speed_tol_q10" in control
    assert "int32_t speed_err_q10" in root_control
    assert "int32_t speed_tol_q10" in root_control


def test_uno_q_rate_limit_rejects_nonpositive_delta() -> None:
    control = (FW / "uno_q_control.h").read_text(encoding="utf-8")
    root_control = (ROOT / "arduino" / "uno_q_control.h").read_text(encoding="utf-8")

    assert "if (max_delta_q10 <= 0)" in control
    assert "if (max_delta_q10 <= 0)" in root_control


def test_air56_unoq_production_port_declares_real_foc_contract() -> None:
    port = (FW / "air56_unoq_hw_port.h").read_text(encoding="utf-8")
    required = [
        "air56_foc_get_omega_meas_rad_s",
        "air56_foc_get_omega_ref_rad_s",
        "air56_foc_get_id_amp",
        "air56_foc_get_iq_amp",
        "air56_foc_get_vdc_volt",
        "air56_foc_get_irms_amp",
        "air56_foc_get_pin_watt",
        "air56_foc_get_status_bits",
        "air56_foc_set_id_ref_amp",
    ]
    for symbol in required:
        assert symbol in port


def test_air56_unoq_platformio_target_exists() -> None:
    pio = (FW / "platformio.ini").read_text(encoding="utf-8")
    assert "board = disco_b_u585i_iot02a" in pio
    assert "air56_unoq_stm32u585_mock" in pio
    assert "air56_unoq_stm32u585_port" in pio
    assert "-DAIR56_UNOQ_USE_MOCK_HW=1" in pio


def test_uno_q_protocol_headers_lock_binary_abi_sizes() -> None:
    fw_protocol = (FW / "uno_q_protocol.h").read_text(encoding="utf-8")
    root_protocol = (ROOT / "arduino" / "uno_q_protocol.h").read_text(encoding="utf-8")
    for text in (fw_protocol, root_protocol):
        assert "sizeof(unoq_telemetry_t) == 20" in text
        assert "sizeof(unoq_command_t) == 9" in text


def test_uno_q_protocol_headers_neutralize_non_finite_sensor_values() -> None:
    fw_protocol = (FW / "uno_q_protocol.h").read_text(encoding="utf-8")
    root_protocol = (ROOT / "arduino" / "uno_q_protocol.h").read_text(encoding="utf-8")
    for text in (fw_protocol, root_protocol):
        assert "if (!isfinite(value))" in text
        assert "return 0;" in text
        assert "return 0u;" in text


def test_air56_unoq_linux_deploy_is_env_based() -> None:
    service = (PKG / "linux" / "air56_unoq_bridge.service").read_text(encoding="utf-8")
    assert "EnvironmentFile=-/etc/default/air56_unoq_bridge" in service
    assert 'cd "$MIC_THEORY_ROOT"' in service
    assert '--serial-port "$SERIAL_PORT"' in service
    assert '--config "$CONFIG_PATH"' in service
    assert "ExecStart=/usr/bin/python3 /opt/mic_theory" not in service


def test_air56_unoq_serial_dependency_declared() -> None:
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "pyserial>=3.5" in req


def test_air56_unoq_static_compile_smoke_script_exists() -> None:
    script = ROOT / "tools" / "check_air56_unoq_firmware_static.py"
    text = script.read_text(encoding="utf-8")
    assert "AIR56_UNOQ_USE_MOCK_HW=1" in text
    assert "air56_unoq_example.ino" in text


def test_air56_unoq_adapter_template_is_not_compiled_as_source() -> None:
    template = FW / "air56_unoq_hw_port_template.cpp.example"
    text = template.read_text(encoding="utf-8")
    assert template.suffix == ".example"
    assert "#error" in text
    assert "air56_foc_set_id_ref_amp" in text


def test_air56_unoq_deploy_smoke_entrypoint_exists() -> None:
    script = ROOT / "tools" / "run_air56_unoq_deploy_smoke.py"
    text = script.read_text(encoding="utf-8")
    assert "tools/air56_unoq_stage0_loopback.py" in text
    assert "tests/test_air56_unoq_stage0_loopback.py" in text
