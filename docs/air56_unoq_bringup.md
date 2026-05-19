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
  smoke:
  `python tools/check_air56_unoq_firmware_static.py --mode mock`
  `python tools/check_air56_unoq_firmware_static.py --mode production-port`
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
- Run the host production-port link smoke first:
  `python tools/check_air56_unoq_firmware_static.py --mode production-port`.
- Implement all `air56_foc_*` functions from `air56_unoq_hw_port.h`.
- Start from `air56_unoq_hw_port_template.cpp.example` and replace each
  `#error` block with the real FOC/inverter call.
- Fill `arduino/air56_unoq_ready/hardware_binding.template.json` with the
  real board binding details and point `adapter.source_files` to the real
  adapter source.
- Validate the manifest and adapter source:
  `python tools/air56_unoq_validate_hw_binding.py --manifest <filled hardware binding manifest>`.
- Run FOC with AI disabled and nominal `id_ref`.
- Validate current offsets, current sign, speed sign, `Vdc`, `P_in`, and fault bits.
- Confirm `air56_foc_set_id_ref_amp()` changes the FOC flux reference only through safe limits.

Pass criteria:

- No fake telemetry remains in the motor-connected build.
- The host production-port smoke links without `AIR56_UNOQ_USE_MOCK_HW`; the final board build still must link against the real STM32U585 FOC/inverter project.
- `hardware_binding_ready=true` from `air56_unoq_validate_hw_binding.py`.
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

Machine-check the A/B logs:

```bash
python tools/air56_unoq_analyze_stage4_ab.py \
  --foc-no-load-csv <real_foc_no_load.csv> \
  --foc-load-step-csv <real_foc_load_step.csv> \
  --ai-no-load-csv <real_ai_no_load.csv> \
  --ai-load-step-csv <real_ai_load_step.csv> \
  --max-current-rms-a <air56_safe_current_limit> \
  --out-json .tmp_pytest/stage4_ab_summary.json
```

Required Stage 4 CSV columns:

- `t_ms`
- `omega_meas`
- `omega_ref`
- `p_in`
- `i_rms`
- `guard_fail`
- `fallback_event`
- optional: `thermal_fault`

## Hardware Acceptance Report

After the real board runs pass, build the report from the recorded Stage 0-4
logs. The checked-in files under `hardware_logs_template/` are examples only;
they are fail-safe and must be replaced with real board evidence.

```bash
python tools/air56_unoq_build_hardware_report.py \
  --board-id unoq-air56-bench-001 \
  --operator bench \
  --stage0-json arduino/air56_unoq_ready/hardware_logs_template/stage0_loopback.json \
  --stage1-json arduino/air56_unoq_ready/hardware_logs_template/stage1_foc_only.json \
  --stage2-json arduino/air56_unoq_ready/hardware_logs_template/stage2_telemetry.json \
  --stage2-csv arduino/air56_unoq_ready/hardware_logs_template/stage2_telemetry.csv \
  --stage3-json arduino/air56_unoq_ready/hardware_logs_template/stage3_ai_tight.json \
  --stage4-json .tmp_pytest/stage4_ab_summary.json \
  --out-json .tmp_pytest/air56_unoq_hardware_acceptance_report.json \
  --out-summary-json .tmp_pytest/air56_unoq_hardware_acceptance_summary_from_logs.json
```

Validate the report:

```bash
python tools/air56_unoq_hardware_acceptance.py \
  --report .tmp_pytest/air56_unoq_hardware_acceptance_report.json \
  --out-json .tmp_pytest/air56_unoq_hardware_acceptance_summary.json
```

`hardware_ready=true` is required before the root plan can call AIR56 UNO Q board deployment `100%` complete.

Required log semantics:

- Stage 0 JSON must prove telemetry `20` bytes, command `9` bytes, CRC error rejection, `>=600 s` loopback, max telemetry period `<=12 ms`, and fallback `<=100 ms`.
- Stage 1 JSON must prove production build without mock, current/speed/Vdc/P_in scaling, fault bits, and safe-disable path.
- Stage 2 JSON/CSV must prove telemetry-only bridge operation with AI disabled; when CSV includes Linux and `stm_*` columns, the builder computes period and decoded mismatch automatically.
- Stage 3 JSON must prove AI enabled only under tight `id_ref` limits with `disable-on-fault` and fallback `<=100 ms`.
- Stage 4 JSON should be generated by `air56_unoq_analyze_stage4_ab.py`; it must document physical FOC baseline vs MIC/AI A/B evidence with no power/guard/tracking/thermal/fallback regression.

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
