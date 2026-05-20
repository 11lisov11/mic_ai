# AIR56 + Delta MS300 VFD ready package

This package is the practical deploy path for a ready-made Delta MS300 inverter:

```text
PC or QRB2210/Linux -> isolated USB-RS485 -> Delta MS300 VFD -> AIR56 motor
```

It is not the STM32U585 FOC/id_ref path. With a commercial VFD the drive owns the
fast current/vector loops; MIC/AI can only supervise safe frequency/run commands
and log telemetry unless the drive exposes deeper flux/current controls.

## Safety contract

- Default config is read-only: `safety.allow_write=false`, `safety.allow_run=false`.
- Frequency writes require both `safety.allow_write=true` in the JSON config and
  CLI flag `--allow-write`.
- Motor start requires both `safety.allow_run=true` in the JSON config and CLI flag
  `--enable-run-command`.
- Use an isolated USB-RS485 adapter. Do not connect the PC to any power terminals.
- First physical run must be no-load, low frequency, with a reachable hardware stop.

## Files

- `config/vfd_delta_ms300_air56.json`: canonical safe default config.
- `vfd/delta_ms300_air56_ready/delta_ms300_air56.config.example.json`: copy/edit for a real bench.
- `tools/delta_ms300_modbus.py`: protocol, safety gate, CLI.
- `tools/delta_ms300_modbus_bridge.py`: short CLI wrapper.
- `docs/delta_ms300_air56_bringup.md`: staged bring-up protocol.
- `vfd/delta_ms300_air56_ready/linux/delta_ms300_air56_monitor.service`: read-only monitor unit.
- `vfd/delta_ms300_air56_ready/windows/run_delta_ms300_stage0.ps1`: Windows smoke helper.

## Minimal command sequence

Dry-run, no hardware:

```powershell
python tools/delta_ms300_modbus_bridge.py --dry-run self-check
python tools/delta_ms300_modbus_bridge.py --dry-run read-once
python tools/delta_ms300_modbus_bridge.py --dry-run stage0 --probe-frequency-hz 1.0
```

Read-only real VFD check:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json self-check
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --out-json .tmp_pytest/delta_ms300_read_once.json read-once
```

Frequency write without motor start, after setting `safety.allow_write=true`:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write --csv-log .tmp_pytest/delta_ms300_frequency_probe.csv set-frequency --hz 5.0 --settle-s 0.2
```

Guarded no-load start, only after wiring and VFD parameters are verified and
`safety.allow_run=true` is set:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write --enable-run-command run-forward
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write stop
```

## Required VFD setup before write/run

Verify these against your MS300 manual revision and keypad menu:

- Frequency command source must be communication / RS-485.
- Operation command source must be communication / RS-485.
- Modbus slave id must match `serial.slave_id`.
- Baud, parity, stop bits must match `serial.baud`, `serial.parity`, `serial.stopbits`.
- Motor nameplate data must be entered into the drive before any AI/MIC comparison.
- Accel/decel ramps must be conservative for the first run.

## What counts as ready

Repo-side ready means dry-run, parser, CRC, safety gates, and monitor tooling pass
automated tests. Hardware-ready means a real Stage 0-4 log exists and shows safe
communication, frequency command response, no-load start/stop, and stable loaded
A/B telemetry. The checked-in defaults intentionally do not start a motor.
