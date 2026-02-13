from __future__ import annotations

"""UNO Q binary protocol helpers (STM <-> Linux).

This module provides fixed-point pack/unpack helpers matching docs/uno_q_deploy.md.
"""

from dataclasses import dataclass
import struct
from typing import ClassVar


OMEGA_SCALE = 1024.0  # rad/s -> q10
CURRENT_SCALE = 1024.0  # A -> q10
VDC_SCALE = 256.0  # V -> q8
POWER_SCALE = 4.0  # W -> q2

CRC16_POLY = 0x1021
CRC16_INIT = 0xFFFF


TELEMETRY_STRUCT = struct.Struct("<IhhhhHhhH")
CMD_STRUCT = struct.Struct("<IBhH")


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
            int(self.t_ms),
            int(round(self.omega_meas * OMEGA_SCALE)),
            int(round(self.omega_ref * OMEGA_SCALE)),
            int(round(self.i_d * CURRENT_SCALE)),
            int(round(self.i_q * CURRENT_SCALE)),
            int(round(self.v_dc * VDC_SCALE)),
            int(round(self.i_rms * CURRENT_SCALE)),
            int(round(self.p_in * POWER_SCALE)),
            int(self.status),
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "Telemetry":
        values = cls._struct.unpack(payload)
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
            int(self.t_ms),
            int(self.enable_ai) & 0xFF,
            int(round(self.id_ref * CURRENT_SCALE)),
            int(self.crc) & 0xFFFF,
        )

    def pack_with_crc(self, init: int = CRC16_INIT) -> bytes:
        payload = self._struct.pack(
            int(self.t_ms),
            int(self.enable_ai) & 0xFF,
            int(round(self.id_ref * CURRENT_SCALE)),
            0,
        )
        crc = crc16_ccitt(payload, init=init)
        return self._struct.pack(
            int(self.t_ms),
            int(self.enable_ai) & 0xFF,
            int(round(self.id_ref * CURRENT_SCALE)),
            int(crc) & 0xFFFF,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "Command":
        values = cls._struct.unpack(payload)
        return cls(
            t_ms=int(values[0]),
            enable_ai=int(values[1]),
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
