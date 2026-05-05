from __future__ import annotations

"""UNO Q binary protocol helpers (STM <-> Linux).

This module provides fixed-point pack/unpack helpers matching docs/uno_q_deploy.md.
"""

from dataclasses import dataclass
import math
import struct
from typing import ClassVar


OMEGA_SCALE = 128.0  # rad/s -> fixed point, int16 range covers AIR56 nominal speed
CURRENT_SCALE = 1024.0  # A -> q10
VDC_SCALE = 256.0  # V -> q8
POWER_SCALE = 4.0  # W -> q2

CRC16_POLY = 0x1021
CRC16_INIT = 0xFFFF


TELEMETRY_STRUCT = struct.Struct("<IhhhhHhhH")
CMD_STRUCT = struct.Struct("<IBhH")


def _finite_float(value: float, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite: {out}")
    return out


def _pack_i16(value: float, field: str) -> int:
    out = int(round(_finite_float(value, field)))
    if out < -32768 or out > 32767:
        raise ValueError(f"{field} out of int16 protocol range: {out}")
    return out


def _pack_u16(value: float, field: str) -> int:
    out = int(round(_finite_float(value, field)))
    if out < 0 or out > 65535:
        raise ValueError(f"{field} out of uint16 protocol range: {out}")
    return out


def _pack_u32(value: float, field: str) -> int:
    out = int(round(_finite_float(value, field)))
    if out < 0 or out > 0xFFFFFFFF:
        raise ValueError(f"{field} out of uint32 protocol range: {out}")
    return out


def _pack_u8(value: float, field: str) -> int:
    out = int(round(_finite_float(value, field)))
    if out < 0 or out > 255:
        raise ValueError(f"{field} out of uint8 protocol range: {out}")
    return out


def _pack_enable_ai(value: float) -> int:
    out = _pack_u8(value, "enable_ai")
    if out not in (0, 1):
        raise ValueError(f"enable_ai must be 0 or 1: {out}")
    return out


def _unpack_exact(struct_obj: struct.Struct, payload: bytes, label: str) -> tuple:
    if len(payload) != struct_obj.size:
        raise ValueError(f"{label} payload must be {struct_obj.size} bytes, got {len(payload)}")
    return struct_obj.unpack(payload)


def crc16_ccitt(payload: bytes, init: int = CRC16_INIT) -> int:
    """CRC-16/CCITT-FALSE over the full packet (CRC field zeroed)."""
    crc = init & 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


@dataclass
class Telemetry:
    t_ms: int
    omega_meas: float
    omega_ref: float
    i_d: float
    i_q: float
    v_dc: float
    i_rms: float
    p_in: float
    status: int

    _struct: ClassVar[struct.Struct] = TELEMETRY_STRUCT

    def pack(self) -> bytes:
        return self._struct.pack(
            _pack_u32(self.t_ms, "t_ms"),
            _pack_i16(self.omega_meas * OMEGA_SCALE, "omega_meas"),
            _pack_i16(self.omega_ref * OMEGA_SCALE, "omega_ref"),
            _pack_i16(self.i_d * CURRENT_SCALE, "i_d"),
            _pack_i16(self.i_q * CURRENT_SCALE, "i_q"),
            _pack_u16(self.v_dc * VDC_SCALE, "v_dc"),
            _pack_i16(self.i_rms * CURRENT_SCALE, "i_rms"),
            _pack_i16(self.p_in * POWER_SCALE, "p_in"),
            _pack_u16(self.status, "status"),
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "Telemetry":
        values = _unpack_exact(cls._struct, payload, "Telemetry")
        return cls(
            t_ms=int(values[0]),
            omega_meas=float(values[1]) / OMEGA_SCALE,
            omega_ref=float(values[2]) / OMEGA_SCALE,
            i_d=float(values[3]) / CURRENT_SCALE,
            i_q=float(values[4]) / CURRENT_SCALE,
            v_dc=float(values[5]) / VDC_SCALE,
            i_rms=float(values[6]) / CURRENT_SCALE,
            p_in=float(values[7]) / POWER_SCALE,
            status=int(values[8]),
        )


@dataclass
class Command:
    t_ms: int
    enable_ai: int
    id_ref: float
    crc: int = 0

    _struct: ClassVar[struct.Struct] = CMD_STRUCT

    def pack(self) -> bytes:
        return self._struct.pack(
            _pack_u32(self.t_ms, "t_ms"),
            _pack_enable_ai(self.enable_ai),
            _pack_i16(self.id_ref * CURRENT_SCALE, "id_ref"),
            _pack_u16(self.crc, "crc"),
        )

    def pack_with_crc(self, init: int = CRC16_INIT) -> bytes:
        payload = self._struct.pack(
            _pack_u32(self.t_ms, "t_ms"),
            _pack_enable_ai(self.enable_ai),
            _pack_i16(self.id_ref * CURRENT_SCALE, "id_ref"),
            0,
        )
        crc = crc16_ccitt(payload, init=init)
        return self._struct.pack(
            _pack_u32(self.t_ms, "t_ms"),
            _pack_enable_ai(self.enable_ai),
            _pack_i16(self.id_ref * CURRENT_SCALE, "id_ref"),
            int(crc) & 0xFFFF,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "Command":
        values = _unpack_exact(cls._struct, payload, "Command")
        enable_ai = int(values[1])
        if enable_ai not in (0, 1):
            raise ValueError(f"enable_ai must be 0 or 1: {enable_ai}")
        return cls(
            t_ms=int(values[0]),
            enable_ai=enable_ai,
            id_ref=float(values[2]) / CURRENT_SCALE,
            crc=int(values[3]),
        )


__all__ = [
    "Telemetry",
    "Command",
    "TELEMETRY_STRUCT",
    "CMD_STRUCT",
    "OMEGA_SCALE",
    "CURRENT_SCALE",
    "VDC_SCALE",
    "POWER_SCALE",
    "CRC16_POLY",
    "CRC16_INIT",
    "crc16_ccitt",
]
