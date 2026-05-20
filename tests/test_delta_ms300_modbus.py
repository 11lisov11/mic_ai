from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import delta_ms300_modbus
from tools.delta_ms300_modbus import (
    COMMAND_RUN_FWD,
    COMMAND_STOP,
    MODBUS_CRC_INIT,
    DeltaMS300Config,
    DeltaMS300Drive,
    DeltaMS300SafetyError,
    DryRunDeltaMS300Client,
    ModbusRtuClient,
    ModbusRtuError,
    SafetyConfig,
    SerialConfig,
    append_crc,
    build_read_holding_frame,
    build_startup_self_check,
    build_write_single_frame,
    clamp_frequency_hz,
    crc16_modbus,
    frequency_to_register,
    load_config,
    parse_read_holding_response,
    parse_register,
    parse_write_single_response,
    register_to_scaled,
    run_stage0_check,
    verify_crc,
)


class EchoTransport:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.responses: list[bytes] = []
        self.closed = False
        self.flushed = False

    def write(self, payload: bytes):
        self.writes.append(payload)
        if payload[1] == 0x03:
            slave = payload[0]
            qty = payload[5]
            data = []
            for index in range(qty):
                value = 0x1200 + index
                data.extend([(value >> 8) & 0xFF, value & 0xFF])
            self.responses.append(append_crc(bytes([slave, 0x03, len(data), *data])))
        elif payload[1] == 0x06:
            self.responses.append(payload)
        return len(payload)

    def read(self, size: int) -> bytes:
        if not self.responses:
            return b""
        payload = self.responses[0]
        chunk = payload[:size]
        self.responses[0] = payload[size:]
        if not self.responses[0]:
            self.responses.pop(0)
        return chunk

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def _armed_cfg() -> DeltaMS300Config:
    return DeltaMS300Config(safety=SafetyConfig(allow_write=True, allow_run=True, max_delta_hz_per_s=100000.0))


def test_crc16_modbus_known_vector_and_frame_helpers() -> None:
    payload = bytes.fromhex("01030000000A")
    assert crc16_modbus(payload) == 0xCDC5
    assert append_crc(payload) == bytes.fromhex("01030000000AC5CD")
    assert verify_crc(bytes.fromhex("01030000000AC5CD"))
    assert not verify_crc(bytes.fromhex("01030000000AC5CE"))
    assert MODBUS_CRC_INIT == 0xFFFF


def test_build_frames_validate_ranges() -> None:
    assert build_read_holding_frame(1, 0x2000, 1)[:6] == bytes.fromhex("010320000001")
    assert build_write_single_frame(1, 0x2001, 500)[:6] == bytes.fromhex("0106200101F4")

    with pytest.raises(ValueError, match="slave_id"):
        build_read_holding_frame(0, 0x2000, 1)
    with pytest.raises(ValueError, match="quantity"):
        build_read_holding_frame(1, 0x2000, 0)
    with pytest.raises(ValueError, match="register"):
        build_write_single_frame(1, 0x10000, 0)


def test_parse_read_and_write_responses() -> None:
    read = append_crc(bytes.fromhex("01030412345678"))
    assert parse_read_holding_response(read, slave_id=1, quantity=2) == [0x1234, 0x5678]
    parse_write_single_response(build_write_single_frame(1, 0x2001, 123), slave_id=1, register=0x2001, value=123)

    with pytest.raises(ModbusRtuError, match="CRC"):
        parse_read_holding_response(read[:-1] + b"\x00", slave_id=1, quantity=2)
    with pytest.raises(ModbusRtuError, match="exception"):
        parse_read_holding_response(append_crc(bytes([1, 0x83, 2])), slave_id=1, quantity=2)
    with pytest.raises(ModbusRtuError, match="echo"):
        parse_write_single_response(build_write_single_frame(1, 0x2001, 124), slave_id=1, register=0x2001, value=123)


def test_modbus_client_roundtrip_uses_transport() -> None:
    transport = EchoTransport()
    client = ModbusRtuClient(transport, slave_id=1, timeout_s=0.2)
    assert client.read_holding(0x2103, 2) == [0x1200, 0x1201]
    client.write_single(0x2001, 500)
    client.close()

    assert transport.flushed
    assert transport.closed
    assert len(transport.writes) == 2


def test_modbus_client_times_out_on_short_response() -> None:
    class EmptyTransport(EchoTransport):
        def write(self, payload: bytes):
            self.writes.append(payload)
            return len(payload)

    client = ModbusRtuClient(EmptyTransport(), slave_id=1, timeout_s=0.001)
    with pytest.raises(ModbusRtuError, match="serial timeout"):
        client.read_holding(0x2000, 1)


def test_config_loads_hex_registers_and_validates(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "schema": delta_ms300_modbus.SCHEMA,
                "drive_model": "VFD4A8MS21ANSAA",
                "motor_name": "AIR56",
                "serial": {"port": "COM9", "baud": 19200, "bytesize": 8, "parity": "E", "stopbits": 1, "timeout_s": 0.2, "slave_id": 2},
                "safety": {"allow_write": False, "allow_run": False},
                "registers": {"command": "2000H", "frequency_command": "0x2001"},
                "scales": {"frequency_hz": 100.0, "current_a": 100.0, "voltage_v": 10.0, "power_kw": 100.0},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.serial.port == "COM9"
    assert cfg.serial.slave_id == 2
    assert cfg.registers.command == 0x2000
    assert parse_register("210Fh", "reg") == 0x210F

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(bad_path)


def test_config_rejects_unsafe_ranges() -> None:
    with pytest.raises(ValueError, match="frequency window"):
        DeltaMS300Config(safety=SafetyConfig(max_frequency_hz=0.0)).validate()
    with pytest.raises(ValueError, match="slave id"):
        DeltaMS300Config(serial=SerialConfig(slave_id=0)).validate()


def test_frequency_scaling_and_clamp() -> None:
    cfg = DeltaMS300Config()
    assert clamp_frequency_hz(60.0, cfg.safety) == 50.0
    assert clamp_frequency_hz(-1.0, cfg.safety) == 0.0
    assert frequency_to_register(12.34, cfg.scales) == 1234
    assert register_to_scaled(1234, cfg.scales.frequency_hz) == 12.34
    with pytest.raises(ValueError, match="finite"):
        frequency_to_register(float("nan"), cfg.scales)


def test_drive_read_snapshot_and_frequency_write() -> None:
    cfg = _armed_cfg()
    dry = DryRunDeltaMS300Client(cfg)
    dry.registers[cfg.registers.output_frequency] = 2500
    dry.registers[cfg.registers.output_current] = 123
    drive = DeltaMS300Drive(cfg, dry)

    snapshot = drive.read_snapshot()
    assert snapshot.output_frequency_hz == 25.0
    assert snapshot.output_current_a == 1.23

    commanded = drive.force_frequency_hz_for_stage0(5.0, armed=True)
    assert commanded == 5.0
    assert dry.registers[cfg.registers.frequency_command] == 500

    drive2 = DeltaMS300Drive(cfg, DryRunDeltaMS300Client(cfg))
    assert drive2.set_frequency_hz(5.0, armed=True) == 5.0


def test_drive_blocks_writes_and_run_until_double_armed() -> None:
    cfg = DeltaMS300Config()
    drive = DeltaMS300Drive(cfg, DryRunDeltaMS300Client(cfg))

    with pytest.raises(DeltaMS300SafetyError, match="frequency write is blocked"):
        drive.set_frequency_hz(5.0, armed=True)
    with pytest.raises(DeltaMS300SafetyError, match="stop write is blocked"):
        drive.stop(armed=True)
    with pytest.raises(DeltaMS300SafetyError, match="run write is blocked"):
        drive.run_forward(armed_write=True, armed_run=True)

    cfg_write_only = DeltaMS300Config(safety=SafetyConfig(allow_write=True, allow_run=False))
    drive = DeltaMS300Drive(cfg_write_only, DryRunDeltaMS300Client(cfg_write_only))
    with pytest.raises(DeltaMS300SafetyError, match="run command is blocked"):
        drive.run_forward(armed_write=True, armed_run=True)


def test_drive_run_and_stop_write_expected_words() -> None:
    cfg = _armed_cfg()
    dry = DryRunDeltaMS300Client(cfg)
    drive = DeltaMS300Drive(cfg, dry)
    drive.run_forward(armed_write=True, armed_run=True)
    drive.stop(armed=True)

    assert dry.writes[-2:] == [(cfg.registers.command, COMMAND_RUN_FWD), (cfg.registers.command, COMMAND_STOP)]


def test_startup_self_check_records_arming_state() -> None:
    cfg = _armed_cfg()
    report = build_startup_self_check(cfg, dry_run=True, allow_write=True, enable_run_command=False)
    assert report["write_armed"] is True
    assert report["run_armed"] is False


def test_stage0_check_read_only_and_write_probe() -> None:
    cfg = _armed_cfg()
    drive = DeltaMS300Drive(cfg, DryRunDeltaMS300Client(cfg))
    report = run_stage0_check(drive, allow_write=False, probe_frequency_hz=1.0)
    assert report["passed"] is True
    assert report["write_probe_enabled"] is False

    report = run_stage0_check(drive, allow_write=True, probe_frequency_hz=1.0)
    assert report["passed"] is True
    assert report["wrote_frequency_hz"] == 1.0


def test_serial_transport_uses_pyserial_contract(monkeypatch) -> None:
    created = []

    class FakeSerial:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            created.append(self)

        def write(self, payload: bytes):
            return len(payload)

        def read(self, size: int) -> bytes:
            return b"\x00" * size

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=FakeSerial))
    transport = delta_ms300_modbus.SerialModbusTransport(SerialConfig(port="COM7", baud=19200, parity="E", stopbits=1))
    transport.write(b"x")
    transport.flush()
    assert transport.read(2) == b"\x00\x00"
    transport.close()

    assert created[0].kwargs["port"] == "COM7"
    assert created[0].kwargs["baudrate"] == 19200
    assert created[0].kwargs["parity"] == "E"
    assert created[0].closed


def test_cli_dry_run_read_once_and_outputs(tmp_path: Path, monkeypatch, capsys) -> None:
    out_json = tmp_path / "read.json"
    csv_log = tmp_path / "read.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_ms300_modbus_bridge.py",
            "--dry-run",
            "--out-json",
            str(out_json),
            "--csv-log",
            str(csv_log),
            "read-once",
        ],
    )

    assert delta_ms300_modbus.main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["schema"] == "mic_theory.delta_ms300.read_once.v1"
    assert "dc_bus_v" in csv_log.read_text(encoding="utf-8")
    assert "read_once" in capsys.readouterr().out


def test_cli_monitor_samples_writes_csv(tmp_path: Path, monkeypatch) -> None:
    csv_log = tmp_path / "monitor.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_ms300_modbus_bridge.py",
            "--dry-run",
            "--csv-log",
            str(csv_log),
            "monitor",
            "--samples",
            "2",
            "--period-s",
            "0.001",
        ],
    )
    assert delta_ms300_modbus.main() == 0
    assert len(csv_log.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_default_blocks_dry_run_write_probe_without_config_arm(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["delta_ms300_modbus_bridge.py", "--dry-run", "--allow-write", "stage0", "--write-probe", "--probe-frequency-hz", "1.0"],
    )
    with pytest.raises(DeltaMS300SafetyError):
        delta_ms300_modbus.main()


def test_cli_run_forward_with_temp_armed_config(tmp_path: Path, monkeypatch) -> None:
    cfg = _armed_cfg()
    cfg_path = tmp_path / "armed.json"
    payload = as_json = {
        "schema": cfg.schema,
        "drive_model": cfg.drive_model,
        "motor_name": cfg.motor_name,
        "serial": cfg.serial.__dict__,
        "safety": cfg.safety.__dict__,
        "registers": cfg.registers.__dict__,
        "scales": cfg.scales.__dict__,
        "command_words": cfg.command_words.__dict__,
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_ms300_modbus_bridge.py",
            "--config",
            str(cfg_path),
            "--dry-run",
            "--allow-write",
            "--enable-run-command",
            "run-forward",
        ],
    )
    assert as_json["safety"]["allow_run"] is True
    assert delta_ms300_modbus.main() == 0


def test_deploy_package_files_exist_and_are_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "config" / "vfd_delta_ms300_air56.json").is_file()
    assert (root / "vfd" / "delta_ms300_air56_ready" / "README.md").is_file()
    assert (root / "docs" / "delta_ms300_air56_bringup.md").is_file()
    assert (root / "vfd" / "delta_ms300_air56_ready" / "linux" / "delta_ms300_air56_monitor.service").is_file()
    cfg = json.loads((root / "config" / "vfd_delta_ms300_air56.json").read_text(encoding="utf-8"))
    assert cfg["safety"]["allow_write"] is False
    assert cfg["safety"]["allow_run"] is False
    text = (root / "vfd" / "delta_ms300_air56_ready" / "README.md").read_text(encoding="utf-8")
    assert "--enable-run-command" in text
    assert "isolated USB-RS485" in text
    service = (root / "vfd" / "delta_ms300_air56_ready" / "linux" / "delta_ms300_air56_monitor.service").read_text(encoding="utf-8")
    assert '--csv-log "$CSV_LOG" "$MODE"' in service


def test_smoke_runner_dry_run(monkeypatch, tmp_path: Path) -> None:
    from tools import run_delta_ms300_deploy_smoke

    out_json = tmp_path / "smoke.json"
    monkeypatch.setattr(sys, "argv", ["run_delta_ms300_deploy_smoke.py", "--dry-run", "--out-json", str(out_json)])
    assert run_delta_ms300_deploy_smoke.main() == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert len(report["steps"]) >= 4
