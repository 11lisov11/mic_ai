# AIR56 UNO Q Bring-Up Protocol

This protocol is the required path before calling the AIR56 UNO Q hardware
deployment complete. It keeps fast control on STM32U585 and uses QRB2210/Linux
only as an AI `id_ref` decision layer.

## Stage 0: Protocol Loopback, No Motor

Goal: prove binary transport before any inverter output.

Checks:

- Build the mock firmware target:
  `pio run -d arduino/air56_unoq_ready/firmware/air56_unoq_example -e air56_unoq_stm32u585_mock`
- If PlatformIO is unavailable on the workstation, run the host static compile
  smoke: `python tools/check_air56_unoq_firmware_static.py`.
- Run the protocol self-test:
  `python tools/air56_unoq_stage0_loopback.py --packets 32`.
- Run the combined repo-side smoke:
  `python tools/run_air56_unoq_deploy_smoke.py`.
- Run the bridge in bench mode against a loopback or UART test harness.
- Verify struct sizes: telemetry `20` bytes, command `9` bytes.
- Verify CRC-16/CCITT-FALSE with CRC field zeroed before calculation.
- Verify little-endian fixed-point scaling, including AIR56 nominal speed at
  `rad/s * 128`.
- Force command silence and confirm STM fallback in `<= 100 ms`.

Pass criteria:

- No framing drift over a 10 minute loopback run.
- CRC failures are rejected.
- Fallback command is observed on bridge shutdown or fatal error.

## Stage 1: STM32U585 FOC Only, AI Disabled

Goal: prove the real hardware adapter and FOC layer independently.

Checks:

- Build production target without `AIR56_UNOQ_USE_MOCK_HW`.
- Implement all `air56_foc_*` functions from `air56_unoq_hw_port.h`.
- Start from `air56_unoq_hw_port_template.cpp.example` and replace each
  `#error` block with the real FOC/inverter call.
- Run FOC with AI disabled and nominal `id_ref`.
- Validate current offsets, current sign, speed sign, `Vdc`, `P_in`, and fault bits.
- Confirm `air56_foc_set_id_ref_amp()` changes the FOC flux reference only through safe limits.

Pass criteria:

- No fake telemetry remains in the motor-connected build.
- Speed and current scaling match independent instruments within the configured tolerance.
- Fault bits trip on known-safe injected fault conditions.

## Stage 2: Telemetry Bridge, Commands Disabled

Goal: prove QRB2210 can receive live telemetry without influencing control.

Checks:

- Run `air56_unoq_bridge.py` with serial transport and `--dry-run`.
- Confirm startup self-checks pass for config and checkpoints.
- Log `omega`, `omega_ref`, `id`, `iq`, estimated load, and status.
- Compare Linux decoded values against STM debug values.

Pass criteria:

- Telemetry period remains near `10 ms`.
- Decoded Linux telemetry matches STM values after fixed-point conversion.
- No bridge crash under steady no-load operation.

## Stage 3: AI Enabled With Tight Limits

Goal: allow AI to adjust `id_ref` while STM remains the safety authority.

Checks:

- Use `--crc --disable-on-fault`.
- Start with tight `--id-min` / `--id-max` around base `id_ref`.
- Keep STM speed-error and fault gates enabled.
- Confirm command timeout returns to base FOC.

Pass criteria:

- `enable_ai=0` on fault or timeout.
- `id_ref` is clamped and slew-limited on STM.
- No tracking guardrail regression versus FOC-only baseline.

## Stage 4: AIR56 A/B Acceptance

Goal: prove that physical AIR56 does not regress and preferably follows the
release direction.

Run:

- FOC-only no-load baseline.
- FOC-only load-step baseline.
- MIC/AI no-load run.
- MIC/AI load-step run.

Record:

- mean input power
- speed tracking error and guard failures
- fallback/gate event count
- `id_ref` command range
- current RMS
- estimated/measured efficiency when shaft power is available

Pass criteria:

- No increase in guard failures versus FOC baseline.
- No sustained fallback oscillation.
- AI-enabled run does not exceed configured current/thermal limits.
- AIR56 physical result is documented separately from the simulation release.

## Fast Verification Commands

Use this profile during normal development on weak hardware:

```bash
python -m pytest -q -m "not slow and not hardware"
```

Targeted deploy checks:

```bash
python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py
```

One-command AIR56 UNO Q deploy smoke:

```bash
python tools/run_air56_unoq_deploy_smoke.py
```

Production-critical AIR56 UNO Q coverage gate:

```bash
python tools/check_air56_unoq_coverage_gate.py
```

Full regression is still allowed, but it is not the default inner-loop command:

```bash
python -m pytest -q
```
