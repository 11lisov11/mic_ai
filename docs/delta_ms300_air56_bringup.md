# Delta MS300 AIR56 bring-up protocol

Target drive: Delta MS300 `VFD4A8MS21ANSAA`, 230 V single phase input, 0.75 kW / 1 HP class.
Target motor path: AIR56 through commercial VFD frequency supervision.

## Architecture

```text
MIC_AI host process
  -> Modbus RTU over isolated USB-RS485
  -> Delta MS300 command/frequency registers
  -> VFD internal scalar/vector/current loops
  -> AIR56 motor
```

This path cannot reproduce the STM32U585 `id_ref` actuator directly. The VFD owns
fast current and flux control. The safe productization goal is:

1. command frequency and run/stop safely;
2. log VFD telemetry reproducibly;
3. compare baseline VFD operation against MIC/AI supervisory profiles;
4. never bypass the drive fault/stop path.

## Stage 0: protocol only, no motor run

Goal: prove serial settings, Modbus CRC, endian, register addresses, and timeout.

1. Connect isolated USB-RS485 to the MS300 RS-485 port per the drive manual.
2. Keep motor mechanically safe; do not enable run command yet.
3. Set `safety.allow_write=false`, `safety.allow_run=false`.
4. Run:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json self-check
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --out-json .tmp_pytest/delta_ms300_stage0_read.json read-once
```

Acceptance:

- no serial timeout;
- CRC passes;
- fault/status/frequency registers are readable;
- no run command is sent.

Optional write probe without run:

1. Set `safety.allow_write=true`, keep `safety.allow_run=false`.
2. Run:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write --out-json .tmp_pytest/delta_ms300_stage0_write_probe.json stage0 --write-probe --probe-frequency-hz 1.0
```

Acceptance:

- frequency command register accepts the value;
- motor remains stopped;
- no fault appears.

## Stage 1: VFD-only no-load run

Goal: prove wiring, nameplate, ramps, direction, stop, current, voltage, and fault handling.

1. Enter motor nameplate parameters into the MS300.
2. Use no-load or mechanically safe low-load condition.
3. Set conservative ramps in the VFD.
4. Set `safety.allow_write=true`, `safety.allow_run=true`.
5. Set frequency to 5 Hz:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write --csv-log .tmp_pytest/delta_ms300_stage1_setfreq.csv set-frequency --hz 5.0 --settle-s 0.2
```

6. Start, monitor, then stop:

```powershell
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write --enable-run-command run-forward
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --csv-log .tmp_pytest/delta_ms300_stage1_monitor.csv monitor --samples 20 --period-s 0.5
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json --allow-write stop
```

Acceptance:

- motor rotates in expected direction;
- stop command works;
- output current stays below `safety.current_limit_a`;
- DC bus stays below `safety.dc_bus_limit_v`;
- no fault register is raised.

## Stage 2: baseline VFD profile

Goal: capture reproducible baseline telemetry before any MIC/AI supervisory layer.

Run a fixed frequency profile manually or through repeated `set-frequency` calls
with low `max_delta_hz_per_s`. Save CSV logs under `.tmp_pytest/` or an external
bench log directory.

Acceptance:

- no command gaps;
- measured output frequency tracks command within drive ramp limits;
- thermal/current limits are stable.

## Stage 3: MIC/AI supervisory profile

The existing research AI controls `id_ref`, which is not directly available via
MS300 Modbus. For this VFD path, MIC/AI must be wrapped as a slow supervisory
policy over frequency trims or profile selection only after Stage 0-2 logs prove
safe operation.

Acceptance:

- AI cannot issue run command by itself;
- AI cannot exceed `max_frequency_hz` or `max_delta_hz_per_s`;
- any fault causes stop/fallback, not retries.

## Stage 4: A/B evidence

Compare VFD baseline vs MIC/AI supervisory profile on identical load conditions.
Minimum logs:

- no-load baseline CSV;
- no-load MIC/AI CSV;
- load-step baseline CSV;
- load-step MIC/AI CSV;
- fault/status summary JSON.

Hardware-ready is not claimed until these logs exist and are reviewed.
