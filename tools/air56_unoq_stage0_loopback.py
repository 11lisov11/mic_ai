from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.uno_q_protocol import CMD_STRUCT, CURRENT_SCALE, TELEMETRY_STRUCT, Command, Telemetry, crc16_ccitt


@dataclass(frozen=True)
class LoopbackFrame:
    index: int
    telemetry_t_ms: int
    command_t_ms: int
    enable_ai: int
    id_ref: float
    crc_ok: bool


@dataclass(frozen=True)
class LoopbackReport:
    packets: int
    telemetry_size: int
    command_size: int
    crc_enabled: bool
    timeout_ms: int
    fallback_after_timeout: bool
    frames: list[LoopbackFrame]

    @property
    def passed(self) -> bool:
        return (
            self.packets > 0
            and len(self.frames) == self.packets
            and self.telemetry_size == 20
            and self.command_size == 9
            and all(frame.crc_ok for frame in self.frames)
            and self.fallback_after_timeout
        )


def _crc_ok(payload: bytes) -> bool:
    expected = int.from_bytes(payload[-2:], byteorder="little")
    actual = crc16_ccitt(payload[:-2] + b"\x00\x00")
    return expected == actual


def _telemetry_stream(packets: int, period_ms: int) -> Iterable[Telemetry]:
    for idx in range(int(packets)):
        yield Telemetry(
            t_ms=idx * int(period_ms),
            omega_meas=144.5 + 0.01 * idx,
            omega_ref=144.5,
            i_d=1.35,
            i_q=0.40 + 0.01 * idx,
            v_dc=24.0,
            i_rms=1.45,
            p_in=42.0 + 0.5 * idx,
            status=0,
        )


def run_loopback_selftest(
    *,
    packets: int = 32,
    period_ms: int = 10,
    timeout_ms: int = 100,
    id_ref_base: float = 1.35,
    id_ref_delta: float = -0.05,
    crc: bool = True,
) -> LoopbackReport:
    if int(packets) <= 0:
        raise ValueError("packets must be positive")
    if int(period_ms) <= 0:
        raise ValueError("period_ms must be positive")
    if int(timeout_ms) < 0:
        raise ValueError("timeout_ms must be non-negative")

    frames: list[LoopbackFrame] = []
    for idx, telemetry in enumerate(_telemetry_stream(packets, period_ms)):
        telem_payload = telemetry.pack()
        decoded = Telemetry.unpack(telem_payload)
        id_ref = float(id_ref_base + id_ref_delta)
        cmd = Command(t_ms=decoded.t_ms, enable_ai=1, id_ref=id_ref)
        cmd_payload = cmd.pack_with_crc() if crc else cmd.pack()
        decoded_cmd = Command.unpack(cmd_payload)
        frames.append(
            LoopbackFrame(
                index=idx,
                telemetry_t_ms=decoded.t_ms,
                command_t_ms=decoded_cmd.t_ms,
                enable_ai=decoded_cmd.enable_ai,
                id_ref=decoded_cmd.id_ref,
                crc_ok=_crc_ok(cmd_payload) if crc else True,
            )
        )

    last_t_ms = (packets - 1) * int(period_ms) if packets > 0 else 0
    silence_t_ms = last_t_ms + int(timeout_ms) + int(period_ms)
    fallback = Command(t_ms=silence_t_ms, enable_ai=0, id_ref=float(id_ref_base))
    fallback_payload = fallback.pack_with_crc() if crc else fallback.pack()
    fallback_cmd = Command.unpack(fallback_payload)
    fallback_after_timeout = fallback_cmd.enable_ai == 0 and abs(fallback_cmd.id_ref - id_ref_base) <= 1.0 / CURRENT_SCALE

    return LoopbackReport(
        packets=int(packets),
        telemetry_size=TELEMETRY_STRUCT.size,
        command_size=CMD_STRUCT.size,
        crc_enabled=bool(crc),
        timeout_ms=int(timeout_ms),
        fallback_after_timeout=fallback_after_timeout,
        frames=frames,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AIR56 UNO Q Stage 0 protocol loopback self-test.")
    parser.add_argument("--packets", type=int, default=32)
    parser.add_argument("--period-ms", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=100)
    parser.add_argument("--id-ref-base", type=float, default=1.35)
    parser.add_argument("--id-ref-delta", type=float, default=-0.05)
    parser.add_argument("--no-crc", dest="crc", action="store_false")
    parser.add_argument("--out-json", default="")
    parser.set_defaults(crc=True)
    args = parser.parse_args()

    report = run_loopback_selftest(
        packets=args.packets,
        period_ms=args.period_ms,
        timeout_ms=args.timeout_ms,
        id_ref_base=args.id_ref_base,
        id_ref_delta=args.id_ref_delta,
        crc=bool(args.crc),
    )
    payload = {
        "passed": report.passed,
        **asdict(report),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
