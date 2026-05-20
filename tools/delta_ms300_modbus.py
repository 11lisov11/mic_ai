from __future__ import annotations

"""Delta MS300 Modbus RTU bridge for AIR56 VFD bring-up.

This module is intentionally conservative. It can read the drive and write a
frequency command, but a run command requires both config-side and CLI-side
arming so a test command cannot start a motor by accident.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODBUS_CRC_INIT = 0xFFFF
MODBUS_CRC_POLY = 0xA001

FUNC_READ_HOLDING = 0x03
FUNC_WRITE_SINGLE = 0x06

REG_COMMAND = 0x2000
REG_FREQUENCY_COMMAND = 0x2001
REG_FAULT_CODE = 0x2100
REG_STATUS = 0x2101
REG_OUTPUT_FREQUENCY = 0x2103
REG_OUTPUT_CURRENT = 0x2104
REG_DC_BUS_VOLTAGE = 0x2105
REG_OUTPUT_VOLTAGE = 0x2106
REG_OUTPUT_POWER = 0x210F

COMMAND_STOP = 0x0001
COMMAND_RUN_FWD = 0x0012
COMMAND_RUN_REV = 0x0022
COMMAND_RESET = 0x0002

SCHEMA = "mic_theory.delta_ms300.air56_config.v1"


class Transport(Protocol):
    def write(self, payload: bytes) -> int | None: ...

    def read(self, size: int) -> bytes: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class ModbusRtuError(RuntimeError):
    pass


class DeltaMS300SafetyError(RuntimeError):
    pass


def crc16_modbus(payload: bytes, init: int = MODBUS_CRC_INIT) -> int:
    crc = int(init) & 0xFFFF
    for byte in payload:
        crc ^= int(byte)
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ MODBUS_CRC_POLY
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc & 0xFFFF


def append_crc(payload: bytes) -> bytes:
    crc = crc16_modbus(payload)
    return payload + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def verify_crc(frame: bytes) -> bool:
    return len(frame) >= 4 and crc16_modbus(frame[:-2]) == int.from_bytes(frame[-2:], "little")


def _u16(value: int, label: str) -> int:
    out = int(value)
    if out < 0 or out > 0xFFFF:
        raise ValueError(f"{label} must be uint16, got {out}")
    return out


def _u8(value: int, label: str) -> int:
    out = int(value)
    if out < 1 or out > 247:
        raise ValueError(f"{label} must be a Modbus slave id in [1, 247], got {out}")
    return out


def _finite_float(value: object, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {out}")
    return out


def parse_register(value: object, label: str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            return _u16(int(text, 16), label)
        if text.endswith("h"):
            return _u16(int(text[:-1], 16), label)
        return _u16(int(text, 10), label)
    return _u16(int(value), label)


def build_read_holding_frame(slave_id: int, start_register: int, quantity: int) -> bytes:
    sid = _u8(slave_id, "slave_id")
    reg = _u16(start_register, "start_register")
    qty = int(quantity)
    if qty < 1 or qty > 125:
        raise ValueError(f"quantity must be in [1, 125], got {qty}")
    payload = bytes((sid, FUNC_READ_HOLDING, (reg >> 8) & 0xFF, reg & 0xFF, (qty >> 8) & 0xFF, qty & 0xFF))
    return append_crc(payload)


def build_write_single_frame(slave_id: int, register: int, value: int) -> bytes:
    sid = _u8(slave_id, "slave_id")
    reg = _u16(register, "register")
    val = _u16(value, "value")
    payload = bytes((sid, FUNC_WRITE_SINGLE, (reg >> 8) & 0xFF, reg & 0xFF, (val >> 8) & 0xFF, val & 0xFF))
    return append_crc(payload)


def parse_read_holding_response(frame: bytes, *, slave_id: int, quantity: int) -> list[int]:
    if len(frame) < 5:
        raise ModbusRtuError(f"read response too short: got {len(frame)}")
    if not verify_crc(frame):
        raise ModbusRtuError("read response CRC mismatch")
    if frame[0] != _u8(slave_id, "slave_id"):
        raise ModbusRtuError(f"read response slave mismatch: expected {slave_id}, got {frame[0]}")
    if frame[1] & 0x80:
        if len(frame) != 5:
            raise ModbusRtuError(f"Modbus exception response length mismatch: got {len(frame)}")
        raise ModbusRtuError(f"Modbus exception for function 0x{frame[1] & 0x7F:02X}: 0x{frame[2]:02X}")
    expected_len = 5 + 2 * int(quantity)
    if len(frame) != expected_len:
        raise ModbusRtuError(f"read response length mismatch: expected {expected_len}, got {len(frame)}")
    if frame[1] != FUNC_READ_HOLDING:
        raise ModbusRtuError(f"read response function mismatch: got 0x{frame[1]:02X}")
    byte_count = int(frame[2])
    if byte_count != 2 * int(quantity):
        raise ModbusRtuError(f"read response byte count mismatch: expected {2 * int(quantity)}, got {byte_count}")
    values: list[int] = []
    for offset in range(3, 3 + byte_count, 2):
        values.append((int(frame[offset]) << 8) | int(frame[offset + 1]))
    return values


def parse_write_single_response(frame: bytes, *, slave_id: int, register: int, value: int) -> None:
    if len(frame) != 8:
        raise ModbusRtuError(f"write response length mismatch: expected 8, got {len(frame)}")
    if not verify_crc(frame):
        raise ModbusRtuError("write response CRC mismatch")
    if frame[0] != _u8(slave_id, "slave_id"):
        raise ModbusRtuError(f"write response slave mismatch: expected {slave_id}, got {frame[0]}")
    if frame[1] & 0x80:
        raise ModbusRtuError(f"Modbus exception for function 0x{frame[1] & 0x7F:02X}: 0x{frame[2]:02X}")
    if frame != build_write_single_frame(slave_id, register, value):
        raise ModbusRtuError("write response echo mismatch")


@dataclass(frozen=True)
class SerialConfig:
    port: str = "COM3"
    baud: int = 9600
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 2
    timeout_s: float = 0.3
    slave_id: int = 1

    def validate(self) -> None:
        if not str(self.port).strip():
            raise ValueError("serial.port is required")
        if int(self.baud) <= 0:
            raise ValueError("serial.baud must be positive")
        if int(self.bytesize) not in (7, 8):
            raise ValueError("serial.bytesize must be 7 or 8")
        if str(self.parity).upper() not in ("N", "E", "O"):
            raise ValueError("serial.parity must be N, E, or O")
        if int(self.stopbits) not in (1, 2):
            raise ValueError("serial.stopbits must be 1 or 2")
        timeout = _finite_float(self.timeout_s, "serial.timeout_s")
        if timeout <= 0.0:
            raise ValueError("serial.timeout_s must be positive")
        _u8(int(self.slave_id), "serial.slave_id")


@dataclass(frozen=True)
class SafetyConfig:
    max_frequency_hz: float = 50.0
    min_frequency_hz: float = 0.0
    max_delta_hz_per_s: float = 2.0
    command_timeout_s: float = 0.5
    allow_write: bool = False
    allow_run: bool = False
    stop_on_exit: bool = True
    current_limit_a: float = 4.8
    dc_bus_limit_v: float = 410.0

    def validate(self) -> None:
        lo = _finite_float(self.min_frequency_hz, "safety.min_frequency_hz")
        hi = _finite_float(self.max_frequency_hz, "safety.max_frequency_hz")
        if lo < 0.0:
            raise ValueError("safety.min_frequency_hz must be non-negative")
        if hi <= 0.0 or lo > hi:
            raise ValueError("safety frequency window is invalid")
        if _finite_float(self.max_delta_hz_per_s, "safety.max_delta_hz_per_s") <= 0.0:
            raise ValueError("safety.max_delta_hz_per_s must be positive")
        if _finite_float(self.command_timeout_s, "safety.command_timeout_s") <= 0.0:
            raise ValueError("safety.command_timeout_s must be positive")
        if _finite_float(self.current_limit_a, "safety.current_limit_a") <= 0.0:
            raise ValueError("safety.current_limit_a must be positive")
        if _finite_float(self.dc_bus_limit_v, "safety.dc_bus_limit_v") <= 0.0:
            raise ValueError("safety.dc_bus_limit_v must be positive")


@dataclass(frozen=True)
class RegisterMap:
    command: int = REG_COMMAND
    frequency_command: int = REG_FREQUENCY_COMMAND
    fault_code: int = REG_FAULT_CODE
    status: int = REG_STATUS
    output_frequency: int = REG_OUTPUT_FREQUENCY
    output_current: int = REG_OUTPUT_CURRENT
    dc_bus_voltage: int = REG_DC_BUS_VOLTAGE
    output_voltage: int = REG_OUTPUT_VOLTAGE
    output_power: int = REG_OUTPUT_POWER


@dataclass(frozen=True)
class ScaleMap:
    frequency_hz: float = 100.0
    current_a: float = 100.0
    voltage_v: float = 10.0
    power_kw: float = 100.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if _finite_float(value, f"scales.{name}") <= 0.0:
                raise ValueError(f"scales.{name} must be positive")


@dataclass(frozen=True)
class CommandWords:
    stop: int = COMMAND_STOP
    run_forward: int = COMMAND_RUN_FWD
    run_reverse: int = COMMAND_RUN_REV
    reset: int = COMMAND_RESET


@dataclass(frozen=True)
class DeltaMS300Config:
    schema: str = SCHEMA
    drive_model: str = "VFD4A8MS21ANSAA"
    motor_name: str = "AIR56"
    serial: SerialConfig = field(default_factory=SerialConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    registers: RegisterMap = field(default_factory=RegisterMap)
    scales: ScaleMap = field(default_factory=ScaleMap)
    command_words: CommandWords = field(default_factory=CommandWords)
    required_drive_parameters: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported config schema: {self.schema}")
        if not str(self.drive_model).strip():
            raise ValueError("drive_model is required")
        if not str(self.motor_name).strip():
            raise ValueError("motor_name is required")
        self.serial.validate()
        self.safety.validate()
        self.scales.validate()


def _dataclass_from_dict(cls: type, payload: dict[str, Any]):
    names = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in payload.items() if key in names})


def _register_map_from_dict(payload: dict[str, Any]) -> RegisterMap:
    data = {name: parse_register(value, f"registers.{name}") for name, value in payload.items() if name in RegisterMap.__dataclass_fields__}
    return RegisterMap(**data)


def load_config(path: str | Path) -> DeltaMS300Config:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    cfg = DeltaMS300Config(
        schema=str(payload.get("schema", SCHEMA)),
        drive_model=str(payload.get("drive_model", "VFD4A8MS21ANSAA")),
        motor_name=str(payload.get("motor_name", "AIR56")),
        serial=_dataclass_from_dict(SerialConfig, dict(payload.get("serial", {}))),
        safety=_dataclass_from_dict(SafetyConfig, dict(payload.get("safety", {}))),
        registers=_register_map_from_dict(dict(payload.get("registers", {}))),
        scales=_dataclass_from_dict(ScaleMap, dict(payload.get("scales", {}))),
        command_words=_dataclass_from_dict(CommandWords, dict(payload.get("command_words", {}))),
        required_drive_parameters=dict(payload.get("required_drive_parameters", {})),
    )
    cfg.validate()
    return cfg


def clamp_frequency_hz(value_hz: float, safety: SafetyConfig) -> float:
    value = _finite_float(value_hz, "frequency_hz")
    return float(max(float(safety.min_frequency_hz), min(float(safety.max_frequency_hz), value)))


def frequency_to_register(value_hz: float, scales: ScaleMap) -> int:
    raw = int(round(_finite_float(value_hz, "frequency_hz") * float(scales.frequency_hz)))
    return _u16(raw, "frequency register value")


def register_to_scaled(value: int, scale: float) -> float:
    return float(_u16(value, "register value")) / float(scale)


class SerialModbusTransport:
    def __init__(self, cfg: SerialConfig) -> None:
        cfg.validate()
        try:
            import serial  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("pyserial is required for Delta MS300 serial Modbus") from exc
        self._ser = serial.Serial(
            port=cfg.port,
            baudrate=int(cfg.baud),
            bytesize=int(cfg.bytesize),
            parity=str(cfg.parity).upper(),
            stopbits=int(cfg.stopbits),
            timeout=float(cfg.timeout_s),
        )

    def write(self, payload: bytes) -> int | None:
        return self._ser.write(payload)

    def read(self, size: int) -> bytes:
        return self._ser.read(size)

    def flush(self) -> None:
        self._ser.flush()

    def close(self) -> None:
        self._ser.close()


class ModbusRtuClient:
    def __init__(self, transport: Transport, *, slave_id: int, timeout_s: float = 0.3) -> None:
        self.transport = transport
        self.slave_id = _u8(slave_id, "slave_id")
        self.timeout_s = float(timeout_s)

    def _read_exact(self, size: int) -> bytes:
        deadline = time.monotonic() + max(float(self.timeout_s), 0.001)
        buf = bytearray()
        while len(buf) < size and time.monotonic() <= deadline:
            chunk = self.transport.read(size - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) != size:
            raise ModbusRtuError(f"serial timeout: expected {size} bytes, got {len(buf)}")
        return bytes(buf)

    def read_holding(self, start_register: int, quantity: int = 1) -> list[int]:
        frame = build_read_holding_frame(self.slave_id, start_register, quantity)
        self.transport.write(frame)
        self.transport.flush()
        response = self._read_exact(5 + 2 * int(quantity))
        return parse_read_holding_response(response, slave_id=self.slave_id, quantity=quantity)

    def write_single(self, register: int, value: int) -> None:
        frame = build_write_single_frame(self.slave_id, register, value)
        self.transport.write(frame)
        self.transport.flush()
        response = self._read_exact(8)
        parse_write_single_response(response, slave_id=self.slave_id, register=register, value=value)

    def close(self) -> None:
        self.transport.close()


class DryRunDeltaMS300Client:
    def __init__(self, cfg: DeltaMS300Config) -> None:
        regs = cfg.registers
        self.registers: dict[int, int] = {
            regs.command: cfg.command_words.stop,
            regs.frequency_command: 0,
            regs.fault_code: 0,
            regs.status: 0,
            regs.output_frequency: 0,
            regs.output_current: 0,
            regs.dc_bus_voltage: int(325.0 * cfg.scales.voltage_v),
            regs.output_voltage: 0,
            regs.output_power: 0,
        }
        self.writes: list[tuple[int, int]] = []

    def read_holding(self, start_register: int, quantity: int = 1) -> list[int]:
        return [int(self.registers.get(int(start_register) + offset, 0)) for offset in range(int(quantity))]

    def write_single(self, register: int, value: int) -> None:
        reg = _u16(register, "register")
        val = _u16(value, "value")
        self.registers[reg] = val
        self.writes.append((reg, val))

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class DeltaMS300Snapshot:
    frequency_command_hz: float
    output_frequency_hz: float
    output_current_a: float
    dc_bus_v: float
    output_voltage_v: float
    output_power_kw: float
    status_raw: int
    fault_raw: int


class DeltaMS300Drive:
    def __init__(self, cfg: DeltaMS300Config, client: ModbusRtuClient | DryRunDeltaMS300Client) -> None:
        cfg.validate()
        self.cfg = cfg
        self.client = client
        self._last_frequency_hz = 0.0
        self._last_frequency_time: Optional[float] = None

    def read_snapshot(self) -> DeltaMS300Snapshot:
        regs = self.cfg.registers
        scales = self.cfg.scales
        return DeltaMS300Snapshot(
            frequency_command_hz=register_to_scaled(self.client.read_holding(regs.frequency_command, 1)[0], scales.frequency_hz),
            output_frequency_hz=register_to_scaled(self.client.read_holding(regs.output_frequency, 1)[0], scales.frequency_hz),
            output_current_a=register_to_scaled(self.client.read_holding(regs.output_current, 1)[0], scales.current_a),
            dc_bus_v=register_to_scaled(self.client.read_holding(regs.dc_bus_voltage, 1)[0], scales.voltage_v),
            output_voltage_v=register_to_scaled(self.client.read_holding(regs.output_voltage, 1)[0], scales.voltage_v),
            output_power_kw=register_to_scaled(self.client.read_holding(regs.output_power, 1)[0], scales.power_kw),
            status_raw=int(self.client.read_holding(regs.status, 1)[0]),
            fault_raw=int(self.client.read_holding(regs.fault_code, 1)[0]),
        )

    def set_frequency_hz(self, frequency_hz: float, *, armed: bool) -> float:
        if not bool(self.cfg.safety.allow_write) or not bool(armed):
            raise DeltaMS300SafetyError("frequency write is blocked; set safety.allow_write=true and pass --allow-write")
        target = clamp_frequency_hz(frequency_hz, self.cfg.safety)
        now = time.monotonic()
        if self._last_frequency_time is not None:
            dt_s = max(0.001, now - self._last_frequency_time)
            max_step = float(self.cfg.safety.max_delta_hz_per_s) * dt_s
            delta = target - self._last_frequency_hz
            if abs(delta) > max_step:
                target = self._last_frequency_hz + math.copysign(max_step, delta)
        raw = frequency_to_register(target, self.cfg.scales)
        self.client.write_single(self.cfg.registers.frequency_command, raw)
        self._last_frequency_hz = float(target)
        self._last_frequency_time = now
        return float(target)

    def force_frequency_hz_for_stage0(self, frequency_hz: float, *, armed: bool) -> float:
        if not bool(self.cfg.safety.allow_write) or not bool(armed):
            raise DeltaMS300SafetyError("frequency write is blocked; set safety.allow_write=true and pass --allow-write")
        target = clamp_frequency_hz(frequency_hz, self.cfg.safety)
        self.client.write_single(self.cfg.registers.frequency_command, frequency_to_register(target, self.cfg.scales))
        self._last_frequency_hz = float(target)
        self._last_frequency_time = time.monotonic()
        return float(target)

    def stop(self, *, armed: bool) -> None:
        if not bool(self.cfg.safety.allow_write) or not bool(armed):
            raise DeltaMS300SafetyError("stop write is blocked; set safety.allow_write=true and pass --allow-write")
        self.client.write_single(self.cfg.registers.command, int(self.cfg.command_words.stop))

    def run_forward(self, *, armed_write: bool, armed_run: bool) -> None:
        if not bool(self.cfg.safety.allow_write) or not bool(armed_write):
            raise DeltaMS300SafetyError("run write is blocked; set safety.allow_write=true and pass --allow-write")
        if not bool(self.cfg.safety.allow_run) or not bool(armed_run):
            raise DeltaMS300SafetyError("run command is blocked; set safety.allow_run=true and pass --enable-run-command")
        self.client.write_single(self.cfg.registers.command, int(self.cfg.command_words.run_forward))

    def close(self) -> None:
        self.client.close()


def build_startup_self_check(cfg: DeltaMS300Config, *, dry_run: bool, allow_write: bool, enable_run_command: bool) -> dict[str, object]:
    cfg.validate()
    return {
        "schema": "mic_theory.delta_ms300.startup_self_check.v1",
        "drive_model": cfg.drive_model,
        "motor_name": cfg.motor_name,
        "serial_port": cfg.serial.port,
        "slave_id": cfg.serial.slave_id,
        "dry_run": bool(dry_run),
        "allow_write_config": bool(cfg.safety.allow_write),
        "allow_write_cli": bool(allow_write),
        "allow_run_config": bool(cfg.safety.allow_run),
        "allow_run_cli": bool(enable_run_command),
        "write_armed": bool(cfg.safety.allow_write and allow_write),
        "run_armed": bool(cfg.safety.allow_write and allow_write and cfg.safety.allow_run and enable_run_command),
        "max_frequency_hz": float(cfg.safety.max_frequency_hz),
        "max_delta_hz_per_s": float(cfg.safety.max_delta_hz_per_s),
        "required_drive_parameters": cfg.required_drive_parameters,
    }


def _snapshot_to_row(snapshot: DeltaMS300Snapshot, *, command_frequency_hz: Optional[float] = None) -> dict[str, object]:
    row = asdict(snapshot)
    if command_frequency_hz is not None:
        row["command_frequency_hz"] = float(command_frequency_hz)
    row["unix_time_s"] = round(time.time(), 6)
    return row


def write_csv_log(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows_list for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def make_drive(cfg: DeltaMS300Config, *, dry_run: bool) -> DeltaMS300Drive:
    if dry_run:
        return DeltaMS300Drive(cfg, DryRunDeltaMS300Client(cfg))
    transport = SerialModbusTransport(cfg.serial)
    client = ModbusRtuClient(transport, slave_id=cfg.serial.slave_id, timeout_s=cfg.serial.timeout_s)
    return DeltaMS300Drive(cfg, client)


def run_stage0_check(
    drive: DeltaMS300Drive,
    *,
    allow_write: bool,
    probe_frequency_hz: float,
) -> dict[str, object]:
    before = drive.read_snapshot()
    wrote_frequency = None
    if allow_write:
        wrote_frequency = drive.force_frequency_hz_for_stage0(probe_frequency_hz, armed=True)
    after = drive.read_snapshot()
    passed = True
    checks = {
        "read_fault_register": before.fault_raw >= 0,
        "read_status_register": before.status_raw >= 0,
        "read_frequency_register": before.frequency_command_hz >= 0.0,
        "optional_frequency_write": True if wrote_frequency is None else abs(after.frequency_command_hz - wrote_frequency) <= 0.02,
        "no_fault_after_probe": after.fault_raw == 0,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "schema": "mic_theory.delta_ms300.stage0_modbus.v1",
        "passed": bool(passed),
        "checks": checks,
        "write_probe_enabled": bool(allow_write),
        "probe_frequency_hz": float(probe_frequency_hz),
        "wrote_frequency_hz": wrote_frequency,
        "before": asdict(before),
        "after": asdict(after),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Delta MS300 Modbus RTU bridge for AIR56 VFD bring-up.")
    parser.add_argument("--config", default="config/vfd_delta_ms300_air56.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-write", action="store_true", help="Arms non-run Modbus writes when config also allows them.")
    parser.add_argument("--enable-run-command", action="store_true", help="Arms motor start only when config safety.allow_run is true too.")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--csv-log", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-check")
    sub.add_parser("read-once")

    stage0 = sub.add_parser("stage0")
    stage0.add_argument("--probe-frequency-hz", type=float, default=0.0)
    stage0.add_argument("--write-probe", action="store_true")

    set_freq = sub.add_parser("set-frequency")
    set_freq.add_argument("--hz", type=float, required=True)
    set_freq.add_argument("--settle-s", type=float, default=0.0)

    monitor = sub.add_parser("monitor")
    monitor.add_argument("--period-s", type=float, default=0.5)
    monitor.add_argument("--samples", type=int, default=0, help="0 means run until interrupted.")

    sub.add_parser("stop")
    sub.add_parser("run-forward")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    drive = make_drive(cfg, dry_run=bool(args.dry_run))
    rows: list[dict[str, object]] = []
    result: dict[str, object]

    try:
        if args.command == "self-check":
            result = build_startup_self_check(
                cfg,
                dry_run=bool(args.dry_run),
                allow_write=bool(args.allow_write),
                enable_run_command=bool(args.enable_run_command),
            )
        elif args.command == "read-once":
            snapshot = drive.read_snapshot()
            result = {"schema": "mic_theory.delta_ms300.read_once.v1", "snapshot": asdict(snapshot)}
            rows.append(_snapshot_to_row(snapshot))
        elif args.command == "stage0":
            write_probe_armed = bool(args.write_probe and args.allow_write)
            result = run_stage0_check(drive, allow_write=write_probe_armed, probe_frequency_hz=float(args.probe_frequency_hz))
        elif args.command == "set-frequency":
            commanded = drive.set_frequency_hz(float(args.hz), armed=bool(args.allow_write))
            if float(args.settle_s) > 0.0:
                time.sleep(float(args.settle_s))
            snapshot = drive.read_snapshot()
            result = {
                "schema": "mic_theory.delta_ms300.set_frequency.v1",
                "commanded_frequency_hz": commanded,
                "snapshot": asdict(snapshot),
            }
            rows.append(_snapshot_to_row(snapshot, command_frequency_hz=commanded))
        elif args.command == "monitor":
            period_s = _finite_float(args.period_s, "--period-s")
            if period_s <= 0.0:
                raise ValueError("--period-s must be positive")
            samples = int(args.samples)
            if samples < 0:
                raise ValueError("--samples must be non-negative")
            count = 0
            last_snapshot: Optional[DeltaMS300Snapshot] = None
            while samples == 0 or count < samples:
                last_snapshot = drive.read_snapshot()
                row = _snapshot_to_row(last_snapshot)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                count += 1
                if samples == 0 or count < samples:
                    time.sleep(period_s)
            result = {
                "schema": "mic_theory.delta_ms300.monitor.v1",
                "samples": count,
                "last_snapshot": {} if last_snapshot is None else asdict(last_snapshot),
            }
        elif args.command == "stop":
            drive.stop(armed=bool(args.allow_write))
            snapshot = drive.read_snapshot()
            result = {"schema": "mic_theory.delta_ms300.stop.v1", "stopped": True, "snapshot": asdict(snapshot)}
            rows.append(_snapshot_to_row(snapshot))
        elif args.command == "run-forward":
            drive.run_forward(armed_write=bool(args.allow_write), armed_run=bool(args.enable_run_command))
            snapshot = drive.read_snapshot()
            result = {"schema": "mic_theory.delta_ms300.run_forward.v1", "run_forward": True, "snapshot": asdict(snapshot)}
            rows.append(_snapshot_to_row(snapshot))
        else:  # pragma: no cover
            raise ValueError(f"unknown command: {args.command}")
    except Exception:
        if bool(cfg.safety.stop_on_exit) and bool(args.allow_write):
            try:
                drive.stop(armed=True)
            except Exception:
                pass
        raise
    finally:
        drive.close()

    if str(args.out_json).strip():
        out_json = Path(str(args.out_json)).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if str(args.csv_log).strip():
        write_csv_log(Path(str(args.csv_log)).resolve(), rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("passed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
