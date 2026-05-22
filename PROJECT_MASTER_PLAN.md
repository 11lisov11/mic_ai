# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-05-22`
Repository: `C:\mic_theory`

## Status

The research/release project has a current-code green `3-motor` Step27 baseline:

- `AIR56`
- `AL31`
- `AO2`

Historical strict-verified release package:

- [20260412_postrestore_ai_3motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release)
- [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)
- `verification_ok = true`

Hardware-productization is now tracked separately:

- `AIR56 UNO Q` is the active hardware deployment path.
- Split architecture is fixed: `STM32U585` owns realtime FOC/safety/fallback; `QRB2210/Linux` owns AI `id_ref` decisions.
- Repository-ready as of this plan: firmware adapter contract, mock-only compile target, env-based Linux service, bridge startup checks, bridge fallback command, staged bring-up protocol.
- Additional repo-side hardening: Stage 0 protocol loopback self-test, deploy-smoke runner, and STM32U585 adapter template.
- Hardware acceptance is now machine-checkable with `tools/air56_unoq_validate_hw_binding.py`, `tools/air56_unoq_analyze_stage4_ab.py`, `tools/air56_unoq_build_hardware_report.py`, `tools/air56_unoq_hardware_acceptance.py`, `tools/air56_unoq_hardware_release_gate.py`, and `tools/air56_unoq_package_hardware_release.py`; it still requires the real STM32U585 adapter plus real Stage 0-4 board logs.
- Not yet physically complete: the real STM32U585 FOC/inverter layer must implement `air56_foc_*` symbols and pass board bring-up.
- `Delta MS300 VFD` is now a second active productization path for the user-specified `VFD4A8MS21ANSAA` inverter.
- Delta MS300 architecture is fixed: `PC/QRB2210/Linux` sends guarded Modbus RTU commands through isolated USB-RS485; the MS300 owns fast current/vector loops and motor protection.
- Repository-ready for Delta MS300: safe default config, Modbus RTU CRC/framing helpers, read-only self-check, guarded frequency writes, guarded run/stop, CSV monitor, Linux/Windows wrappers, staged bring-up docs, and automated smoke tests.
- Not yet physically complete for Delta MS300: real USB-RS485 Stage 0, no-load run, baseline logs, MIC/AI supervisory logs, and loaded A/B evidence must be captured on the actual drive/motor.
- A new theory/research branch is active: `Safe Neural Horizon PWM with Event-Triggered Twin Feedback`.
- New branch status: host-level alpha-beta model, two-level inverter vector model, AI-PWM Safety Gateway, horizon controller, event-feedback twin scaffold, tests, and MC=100 smoke are implemented.
- New branch limitation: this is not MCU/HIL/bench evidence and must not be described as hardware-ready.

Historical strict-verified `2-motor` release kept for provenance:

- [20260412_postrestore_ai_2motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release)

## Canonical Research Completion Criteria

The research/release scope counts as `100% done` only when all items below are true.

1. Strict `Step27/Step28` package exists for all three motors.
2. `VERIFY_SUBMISSION_CANDIDATE.json` is green.
3. Per-motor acceptance is green for `AIR56`, `AL31`, and `AO2`.
4. Root documentation reflects the real scope and active release tag.
5. Regression tests are green after the final code changes.

Status on `2026-04-12`:

- all five conditions are satisfied

Status on `2026-05-19` full training refresh:

- current-code strict Step27 selected recheck is green for all three motors
- `AIR56`: `+1.072%` avg power saving, `+0.112%` eta gain, `0` err failures, `envelope_fail_count = 0`
- `AL31`: `+3.455%` avg power saving, `+0.003%` eta gain, `0` err failures, `envelope_fail_count = 0`
- `AO2`: `+0.512%` avg power saving, `+1.724%` eta gain, `0` err failures, `envelope_fail_count = 0`
- strict recheck output: [final_selected_strict_recheck_20260519](C:/mic_theory/outputs/train3_fullprog_20260519/final_selected_strict_recheck_20260519)
- tracked refresh manifest: [20260519_train3_refresh](C:/mic_theory/paper/ieee_2026/data/release/20260519_train3_refresh/research_refresh_manifest.json)
- `AO2` live config was refreshed to the reproducible `ao2_current_repro_rand017` candidate
- `AL31` live config and checkpoint registry now point to the promoted `2026-05-19` `best_actor_step27_train3.pth` checkpoint, selected from `actor_ep018.pth`
- old `outputs/ao2_fw_grid_20260412af/fw_c` is preserved as provenance but is no longer the canonical current-code acceptance source
- Step27 scan resume-state now hashes config, candidate, acceptance envelope, and checkpoint content to prevent stale acceptance reuse
- `train_3motors_pipeline.py --step27-select` full refresh passed `3/3`; `AIR56` and `AO2` kept canonical baselines, `AL31` promoted the new fine-tuned checkpoint

## Hardware Completion Criteria

The `AIR56 UNO Q` board deployment counts as complete only when all items below are true.

1. Production STM32U585 build links without `AIR56_UNOQ_USE_MOCK_HW`.
2. Board code implements all `air56_foc_*` functions from `air56_unoq_hw_port.h`.
3. PlatformIO production-port build or equivalent STM32U585 build passes.
4. Stage 0 loopback passes without framing/CRC drift.
5. Stage 1 STM-only FOC passes with validated current/speed/Vdc/P_in scaling.
6. Stage 2 telemetry-only bridge passes with AI disabled.
7. Stage 3 AI-enabled tight-limit run passes without tracking/fault regression.
8. Stage 4 AIR56 physical A/B run is documented against FOC baseline.

Status on `2026-05-05`:

- repo-side deploy package is implemented
- physical FOC/HAL binding is still an external integration requirement

Status on `2026-05-09` audit:

- repo-side AIR56 UNO Q deploy smoke is green
- weak-hardware fast pytest profile is green
- production-critical AIR56 UNO Q coverage gate is green
- physical board deployment is still not complete because the real STM32U585 FOC/inverter adapter and staged motor tests are not present in this repository

## Delta MS300 VFD Completion Criteria

The `Delta MS300 VFD` AIR56 path counts as hardware-complete only when all items below are true.

1. Read-only Modbus Stage 0 passes on the real isolated USB-RS485 link.
2. Optional frequency-command write probe passes while the motor remains stopped.
3. MS300 motor nameplate, command source, frequency source, and serial parameters are confirmed on the keypad/manual revision.
4. VFD-only no-load run at low frequency passes with safe stop and no fault.
5. Baseline VFD frequency profile is logged to CSV.
6. MIC/AI supervisory frequency/profile layer is enabled with strict frequency and ramp limits.
7. Stage 4 physical A/B logs show no regression in current, DC bus, fault status, tracking, or power guardrails.

Status on `2026-05-20`:

- repo-side Delta MS300 deploy package is implemented
- automated dry-run smoke is green
- physical MS300 drive tests are still open because the real inverter is not connected in this environment

## Safe Neural Horizon PWM Research Criteria

The new `Safe Neural Horizon PWM` research track counts as theory-complete only when all items below are true.

1. Alpha-beta induction motor model covers flux, current, torque, mechanics, parameter drift, saturation hooks, current/voltage limits, dead-time/Vdc effects.
2. Two-level inverter model covers all 8 vectors, common-mode voltage, dead-time, min-pulse, switching/conduction loss proxies, and thermal proxy.
3. Safety Gateway proves and tests no-shoot-through timing waveforms for every vector transition.
4. AI-PWM compares one-step, H=2, H=3, H=4, thermal cost, spectral cost, and feedback economy variants.
5. Neural twin includes domain randomization, multi-step loss protocol, uncertainty/confidence, and online adaptation experiment.
6. Event-triggered feedback is compared with fixed 10 kHz, 5 kHz, 2 kHz, 1 kHz, 500 Hz, 200 Hz, and sensorless/current-only proxies.
7. Strong baselines exist: FOC-SVM, FCS-MPC, DTC, DTC-SVM, deadbeat predictive current control, sensorless/adaptive FOC.
8. Robust scenario matrix covers start, load, reverse, braking, low speed, field weakening, DC-link sag, heating, sensor faults, OOD, and fault injection.
9. Monte Carlo reaches at least `N=100` for first study and `N=500..1000` for final paper candidate.
10. Ablation study and Pareto fronts are generated.
11. Scientific report and article draft state exactly what is host-simulated and what still requires MCU/HIL/bench.

Status on `2026-05-22`:

- host-level scaffold is implemented
- no-shoot-through waveform invariant tests pass for all 8x8 vector transitions
- MC=100 smoke runs and records zero safety waveform violations for the new H=2 variant
- MC=5 scenario matrix runs across start/no-load, start/load, load-step, load-shed, reverse, low-speed, DC-sag, and sensor-dropout host scenarios
- full host matrix release runs across `31` scenarios: the 30-scenario TZ set plus an explicit `sensor_dropout` stress case
- host-level proxy baselines exist for FOC-SVM, FCS-MPC, DTC, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC; these are explicitly not final strong baselines
- ablation and Pareto smoke extraction are implemented
- host-level fault-injection summary is implemented and reports no shoot-through
- tracked host release package exists: [20260522_host_release](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release)
- two bugs were found and fixed: pre-step current reporting in the alpha-beta model, and missing flux-building cost that made startup select zero vector
- strong key-level baselines, full robust matrix, final ablation, Pareto, HIL, and bench work are still open

## Active Rework Plan After 2026-05-09 Audit

This checklist tracks what is still not complete after the research release. It intentionally separates already-closed software/research work from board-connected deployment work.

### A. Repository Hygiene And Publication Artifacts

- [x] Keep the canonical `3-motor` research release as the active scientific baseline.
- [x] Record the `2026-05-19` full training refresh as a tracked release manifest.
- [x] Promote the strict-green `AL31` refresh checkpoint in the live config and checkpoint registry.
- [x] Regenerate AIR56 working-characteristics figure outputs with readable labels and math-style subscripts.
- [x] Rebuild the legacy nominal power plot as a vector PDF/SVG/PNG instead of a raster screenshot export.
- [x] Decide whether the regenerated publication figures are canonical and either commit them or move noncanonical scratch artifacts out of tracked publication folders.
- [x] Rename or remove the misleading `fig_nominal_power_legacy_from_screenshot.*` compatibility files after downstream documents stop referencing them.
- [x] Run `git diff --check` and a final `git status --short --untracked-files=all` before the next push.

### B. AIR56 UNO Q Repo-Side Software Readiness

- [x] Protocol, Linux bridge, Stage 0 loopback, firmware static compile, and deploy smoke pass locally.
- [x] Host production-port firmware link smoke passes without `AIR56_UNOQ_USE_MOCK_HW` when the `air56_foc_*` contract is supplied by a test shim.
- [x] Focused AIR56 UNO Q tests pass locally.
- [x] Weak-hardware fast test profile passes locally.
- [x] AIR56 UNO Q production-critical coverage gate passes locally.
- [x] Increase `tools/air56_unoq_bridge.py` test coverage from the current `68%` toward `75-80%` and raise the coverage gate floor to `75%`.
- [x] Add a small regression check that fails if the root plan reports `100%` while hardware completion items are still open.
- [x] Add a machine-checkable AIR56 UNO Q hardware Stage 0-4 acceptance report validator.
- [x] Add a fail-safe Stage 0-4 log-to-report builder so real board logs become reproducible acceptance evidence instead of hand-edited JSON.
- [x] Add a fail-safe Stage 4 physical A/B analyzer so FOC baseline vs MIC/AI CSV logs are checked for power, tracking, guard, current/thermal, and fallback regressions.
- [x] Add a fail-safe STM32U585 hardware binding validator for board pinout/scaling/fault manifest and production `air56_foc_*` adapter source.
- [x] Add a final fail-safe AIR56 hardware release gate aggregating hardware binding, physical acceptance, deploy smoke, and coverage evidence.
- [x] Add a final AIR56 hardware release evidence packager with SHA-256 hashes for binding, acceptance, smoke, coverage, and gate evidence.

### C. STM32U585 Production Hardware Binding

- [x] Keep the mock adapter separate and explicitly non-production.
- [x] Keep `air56_unoq_hw_port.h` as the production FOC/inverter adapter contract.
- [ ] Implement the real board-side translation unit for all `air56_foc_*` symbols:
  - `air56_foc_get_omega_meas_rad_s`
  - `air56_foc_get_omega_ref_rad_s`
  - `air56_foc_get_id_amp`
  - `air56_foc_get_iq_amp`
  - `air56_foc_get_vdc_volt`
  - `air56_foc_get_irms_amp`
  - `air56_foc_get_pin_watt`
  - `air56_foc_get_status_bits`
  - `air56_foc_set_id_ref_amp`
- [ ] Confirm the actual UNO Q STM32U585 board definition, pinout, UART instance, ADC/current scaling, and inverter enable/fault lines; the current PlatformIO target is a reproducible STM32U585 compile target, not proof of final board pin binding.
- [ ] Build the production target without `AIR56_UNOQ_USE_MOCK_HW` against the real FOC/inverter project.
- [x] Add a repo-side static regression that the production PlatformIO target does not enable `AIR56_UNOQ_USE_MOCK_HW`.
- [x] Add a machine-checkable hardware binding manifest validator that rejects mock/stub adapter source and incomplete pinout/scaling/fault mappings.
- [ ] Verify the final motor-connected board binary and release procedure cannot accidentally use the mock adapter.

### D. Physical Bring-Up And Acceptance

- [x] Stage 0 loopback protocol self-test passes in repo-side simulation.
- [ ] Stage 0 loopback must pass on the actual QRB2210-to-STM32U585 serial link.
- [ ] Stage 1 STM-only FOC must pass without AI: current scaling, speed scaling, Vdc scaling, `P_in` estimate, fault bits, and safe disable path.
- [ ] Stage 2 bridge telemetry-only mode must pass with AI disabled.
- [ ] Stage 3 AI-enabled run must pass with narrow `id_ref` limits, `disable-on-fault`, and fallback within `100 ms`.
- [ ] Stage 4 physical A/B comparison must document AIR56 FOC baseline vs MIC/AI `id_ref` under no-load and load-step conditions.
- [x] Provide checked-in Stage 0-4 log templates and a report generator; checked-in templates intentionally fail until replaced with real board logs.

### E. Release/Verification Discipline

- [x] Do not restart long RL training unless a measured regression proves it is necessary.
- [x] Use existing strict `Step28` release artifacts as the research baseline.
- [x] Add Step27 selection of canonical release baselines so a new training run cannot silently regress below the accepted baseline.
- [x] Add file-hash based Step27 scan resume signatures to prevent stale metric reuse.
- [x] Refresh `AO2` current-code candidate after stale-cache audit and recheck strict Step27 acceptance.
- [x] Run the full `2026-05-19` joint plus fine-tune training refresh and record the selected strict result.
- [ ] Do not call the whole project `100% hardware-ready` until Sections C and D are closed on real hardware.
- [ ] Before final hardware release, run:
  - `python -m pytest -q -m "not slow and not hardware"`
  - `python tools/run_air56_unoq_deploy_smoke.py`
  - `python tools/check_air56_unoq_coverage_gate.py`
  - `python tools/check_air56_unoq_firmware_static.py --mode production-port`
  - `python tools/air56_unoq_validate_hw_binding.py --manifest <filled real hardware binding manifest>`
  - `python tools/air56_unoq_analyze_stage4_ab.py --foc-no-load-csv <foc_no_load.csv> --foc-load-step-csv <foc_load_step.csv> --ai-no-load-csv <ai_no_load.csv> --ai-load-step-csv <ai_load_step.csv> --max-current-rms-a <limit> --out-json <stage4_ab_summary.json>`
  - `python tools/air56_unoq_build_hardware_report.py --board-id <board> --operator <name> --stage0-json <stage0.json> --stage1-json <stage1.json> --stage2-json <stage2.json> --stage2-csv <stage2.csv> --stage3-json <stage3.json> --stage4-json <stage4.json> --out-json <filled real hardware report>`
  - `python tools/air56_unoq_hardware_acceptance.py --report <filled real hardware report>`
  - `python tools/air56_unoq_hardware_release_gate.py --binding-manifest <filled real hardware binding manifest> --hardware-report <filled real hardware report> --deploy-smoke-json <deploy smoke report> --coverage-json <coverage gate json>`
  - `python tools/air56_unoq_package_hardware_release.py --package-tag <tag> --out-dir <release package dir> --binding-manifest <filled real hardware binding manifest> --hardware-report <filled real hardware report> --deploy-smoke-json <deploy smoke report> --coverage-json <coverage gate json>`
  - production firmware build without mock hardware
  - physical Stage 0-4 bring-up protocol

### F. Delta MS300 VFD AIR56 Productization

- [x] Add safe default Delta MS300 AIR56 config with `allow_write=false` and `allow_run=false`.
- [x] Implement Modbus RTU CRC/framing, read holding registers, write single register, and strict response validation.
- [x] Implement safety gates so frequency writes require config and CLI arming.
- [x] Implement safety gates so run commands require separate config and CLI arming.
- [x] Add read-only self-check, Stage 0 probe, monitor, CSV logging, stop, and run-forward CLI modes.
- [x] Add Linux env/service template and Windows Stage 0 helper.
- [x] Add staged Delta MS300 AIR56 bring-up documentation.
- [x] Add repo-side Delta MS300 smoke runner.
- [x] Add targeted Delta MS300 regression tests.
- [ ] Stage 0 read-only Modbus must pass on the actual Delta MS300 USB-RS485 link.
- [ ] Stage 0 optional frequency write probe must pass while the motor remains stopped.
- [ ] MS300 keypad parameters must be confirmed against the real manual revision and saved with bench notes.
- [ ] Stage 1 VFD-only no-load AIR56 run must pass at low frequency with safe stop and no fault.
- [ ] Stage 2 baseline VFD profile must be logged to CSV.
- [ ] Stage 3 MIC/AI supervisory profile must run with strict frequency/ramp limits and no automatic run authority.
- [ ] Stage 4 Delta MS300 physical A/B comparison must document baseline vs MIC/AI supervisory logs.
- [ ] Delta MS300 hardware-ready claim must remain false until real Stage 0-4 evidence exists.

### G. Safe Neural Horizon PWM Research Track

- [x] Add stationary alpha-beta induction motor model for the new theory branch.
- [x] Add two-level inverter vector model with nonideal voltage/loss/common-mode proxies.
- [x] Add protected AI-PWM Safety Gateway with valid vector, min-pulse, current, Vdc, Tj, confidence, risk, watchdog, switching-budget, and fault-latch checks.
- [x] Add host test that all 8x8 vector transitions generate no shoot-through gate waveform.
- [x] Add neural-horizon AI-PWM controller scaffold with H=1..4 search support.
- [x] Add neural-cost-shaping and event-triggered feedback/twin scaffold without claiming trained optimality.
- [x] Add host-level H=4 bounded sequence-selection smoke.
- [x] Add host-level fault-injection tests for invalid vector, min-pulse, confidence, overcurrent, undervoltage, overtemperature, and watchdog.
- [x] Add fast MC smoke runner that writes to `.tmp_pytest/` instead of tracked output paths.
- [x] Run first `N=100` host-level smoke for the new branch.
- [x] Add host-level proxy baselines for FOC-SVM, FCS-MPC, DTC, DTC-SVM, deadbeat current control, and sensorless/adaptive FOC.
- [x] Add scenario matrix smoke for start/no-load, start/load, load-step, load-shed, reverse, low-speed, DC-sag, and sensor-dropout cases.
- [x] Expand scenario matrix to the full TZ host set plus explicit `sensor_dropout` stress case.
- [x] Add ablation smoke variants for horizon, feedback density, switching penalty, and current penalty.
- [x] Add Pareto front extraction for host-level study outputs.
- [x] Add markdown report builder for Safe Neural Horizon PWM JSON outputs.
- [x] Add tracked Safe Neural Horizon PWM host release package with JSON results, markdown report, article draft, open-items file, and SHA-256 manifest.
- [x] Document found bugs and current limitations in [safe_neural_horizon_pwm_research.md](C:/mic_theory/docs/safe_neural_horizon_pwm_research.md).
- [ ] Replace proxy FOC-SVM with a tuned key-level FOC-SVM baseline using the same inverter/dead-time/min-pulse/current constraints.
- [ ] Replace proxy FCS-MPC with a tuned FCS-MPC current/torque/flux baseline.
- [ ] Replace proxy DTC and DTC-SVM with tuned DTC/DTC-SVM baselines.
- [ ] Replace proxy deadbeat current control with tuned deadbeat predictive current-control baseline.
- [ ] Replace proxy sensorless/adaptive FOC with MRAS/EKF/adaptive FOC.
- [x] Expand robust host scenario matrix to the full 30-scenario research TZ plus one explicit dropout stress case.
- [x] Expand fault-injection matrix to include raw shoot-through request emulation, no-dead-time transition emulation, and hardware-like desat/UVLO cases.
- [ ] Run full ablation: gateway/current shield/confidence/switching budget/min-pulse/horizon/thermal/spectral/twin/randomization/feedback variants with publication-scale MC.
- [ ] Generate publication-grade Pareto fronts.
- [ ] Generate publication-grade plots for speed, torque, currents, gates, switching events, feedback events, confidence, losses, temperature, FFT, and Pareto.
- [x] Prepare host-level article draft; clearly mark MCU/HIL/bench as not done.

### 2026-05-20 Delta MS300 Commands

Commands run during this audit:

- `python -m pytest -q tests/test_delta_ms300_modbus.py`
  - result: `20 passed`
- `python tools/run_delta_ms300_deploy_smoke.py --out-json .tmp_pytest/delta_ms300_smoke.json`
  - result: `passed = true`
- `python tools/delta_ms300_modbus_bridge.py --dry-run --csv-log .tmp_pytest/delta_ms300_monitor.csv monitor --samples 2 --period-s 0.001`
  - result: `passed`, CSV monitor output written
- `.venv\Scripts\python.exe -m pytest -q -m "not slow and not hardware"`
  - result: `333 passed, 18 deselected`
- `.venv\Scripts\python.exe -m pytest -q`
  - result: `351 passed`

### 2026-05-09 Audit Commands

Commands run during this audit:

- `python tools/run_air56_unoq_deploy_smoke.py`
  - result: `passed = true`
- `python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py`
  - result: `61 passed`
- `python tools/check_air56_unoq_coverage_gate.py`
  - initial audit result: `passed = true`, total AIR56 deploy subset coverage `79.26%`
  - after bridge coverage hardening: `passed = true`, total AIR56 deploy subset coverage `86.22%`, `tools/air56_unoq_bridge.py = 79.16%`, bridge floor raised to `75%`
  - after train3 refresh implementation: `passed = true`, total AIR56 deploy subset coverage `85.90%`, `tools/air56_unoq_bridge.py = 78.75%`
  - after hardware acceptance validator: `passed = true`, total AIR56 deploy subset coverage `87.08%`, `tools/air56_unoq_hardware_acceptance.py = 98.57%`
  - after production-port link smoke: `passed = true`, total AIR56 deploy subset coverage `87.12%`, `tools/check_air56_unoq_firmware_static.py = 96.67%`
  - after hardware log builder: `passed = true`, total AIR56 deploy subset coverage `89.48%`, `tools/air56_unoq_build_hardware_report.py = 100.00%`
  - after Stage 4 A/B analyzer: `passed = true`, total AIR56 deploy subset coverage `90.70%`, `tools/air56_unoq_analyze_stage4_ab.py = 100.00%`
  - after hardware binding validator: `passed = true`, total AIR56 deploy subset coverage `91.57%`, `tools/air56_unoq_validate_hw_binding.py = 100.00%`
  - after hardware release gate: `passed = true`, total AIR56 deploy subset coverage `92.06%`, `tools/air56_unoq_hardware_release_gate.py = 100.00%`
  - after hardware release packager: `passed = true`, total AIR56 deploy subset coverage `92.42%`, `tools/air56_unoq_package_hardware_release.py = 100.00%`
- `python -m pytest -q -m "not slow and not hardware"`
  - initial audit result: `257 passed, 14 deselected`
  - after repo-side hardening: `270 passed, 14 deselected`
  - after hardware log builder: `290 passed, 18 deselected`
  - after Stage 4 A/B analyzer: `297 passed, 18 deselected`
  - after hardware binding validator: `303 passed, 18 deselected`
  - after hardware release gate: `308 passed, 18 deselected`
  - after hardware release packager: `313 passed, 18 deselected`
- `python -m pytest -q tests/test_air56_unoq_bridge.py tests/test_report_plan_completion_smoke.py`
  - result: `40 passed`
- `python -m pytest -q tests/test_air56_unoq_deploy_package.py tests/test_air56_unoq_bridge.py tests/test_report_plan_completion_smoke.py`
  - result: `55 passed`

Not run during this audit:

- slow-only `python -m pytest -q -m "slow and not hardware"`
- PlatformIO production build against real board FOC/inverter sources
- hardware-marked physical tests
- physical AIR56 Stage 0-4 bring-up

## Final Canonical Artifacts

### Release package

- [FINAL_CHECKLIST_AUTO.md](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/FINAL_CHECKLIST_AUTO.md)
- [SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/SUBMISSION_CANDIDATE.json)
- [IEEE_SUBMISSION_DOSSIER.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/IEEE_SUBMISSION_DOSSIER.json)
- [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)

### Per-motor acceptance

- [motor_tuning_acceptance_summary.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_tuning_acceptance_summary.json)

Confirmed final state:

- `AIR56`: `acceptance_pass = true`
- `AL31`: `acceptance_pass = true`
- `AO2`: `acceptance_pass = true`

### Reproduce outputs

- [step28_reproduce_manifest.json](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/step28_reproduce_manifest.json)
- [mode1 summary](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/mode1_foc_encoder_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv)
- [mode2 summary](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/mode2_foc_sensorless_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv)

## AO2 Closure Record

`AO2` was the last real blocker.

The closure path that worked:

1. Detect the mismatch between the old live runtime config and the motor nameplate.
2. Build a nameplate-first AO2 branch.
3. Confirm that the remaining blocker is `FOC` headroom / saturation, not only PPO quality.
4. Add optional `field_weakening` to the FOC controller.
5. Sweep cheap `field_weakening` parameters on the already tuned AO2 actor.
6. Promote the strict-green configuration into the live AO2 research config and registry.
7. Rebuild the full strict `3-motor` package.

Canonical AO2 artifacts:

- diagnosis tool: [diagnose_motor_nominal_consistency.py](C:/mic_theory/tools/diagnose_motor_nominal_consistency.py)
- preserved nameplate-first config: [env_backlog_ao2_nameplate_first.py](C:/mic_theory/config/env_backlog_ao2_nameplate_first.py)
- tuned AO2 backlog config: [env_backlog_ao2_nameplate_foc_tuned.py](C:/mic_theory/config/env_backlog_ao2_nameplate_foc_tuned.py)
- live AO2 config: [env_research_ao2_32_4_3kw.py](C:/mic_theory/config/env_research_ao2_32_4_3kw.py)
- current-code AO2 candidate: [step27_ao2_current_repro_candidate_20260519.json](C:/mic_theory/config/step27_ao2_current_repro_candidate_20260519.json)
- current-code strict scan: [ao2_current_repro_strict_scan_20260519](C:/mic_theory/outputs/ao2_current_repro_strict_scan_20260519)
- old FW sweep result kept for provenance, not current-code acceptance: [fw_c summary](C:/mic_theory/outputs/ao2_fw_grid_20260412af/fw_c/ao2_checkpoint_scan_summary.json)

## Final Regression Reference

Focused final regression after the AO2 closure work:

- `python -m pytest -q tests/test_step27_report_markdown.py tests/test_step27_hybrid_trigger.py tests/test_vector_foc_field_weakening.py tests/test_scan_step27_checkpoints.py tests/test_train_ai_id_ref_external_step27.py tests/test_diagnose_motor_nominal_consistency.py`
- `60 passed`

Focused regression after the `2026-05-19` training/Step27 reproducibility fixes:

- `python -m pytest -q tests/test_scan_step27_checkpoints.py tests/test_tune_motor_step27_candidates.py tests/test_train_3motors_pipeline_smoke.py tests/test_train_3motors_pipeline_joint_and_finetune_smoke.py tests/test_train_3motors_pipeline_resume_eval_first_smoke.py`
- `34 passed`
- full train3 refresh run:
  - joint domain-randomized seed `101`: complete
  - fine-tune per motor seed `101`: complete
  - selected strict recheck: `3/3` pass
  - tracked manifest: [20260519_train3_refresh](C:/mic_theory/paper/ieee_2026/data/release/20260519_train3_refresh/research_refresh_manifest.json)
- expanded focused regression with AO2 hardening/vector FOC/Step27 external checks:
  - `python -m pytest -q tests/test_scan_step27_checkpoints.py tests/test_tune_motor_step27_candidates.py tests/test_train_3motors_pipeline_smoke.py tests/test_train_3motors_pipeline_joint_and_finetune_smoke.py tests/test_train_3motors_pipeline_resume_eval_first_smoke.py tests/test_ao2_hardening_sweep_smoke.py tests/test_vector_foc_field_weakening.py tests/test_step27_report_markdown.py tests/test_train_ai_id_ref_external_step27.py`
  - `67 passed`

Full repository regression must remain green before final push:

- `python -m pytest -q`
- current `2026-05-19` result after train3 refresh implementation: `292 passed`
- current `2026-05-19` result after hardware acceptance validator: `300 passed`
- current `2026-05-19` result after production-port link smoke: `301 passed`
- current `2026-05-19` result after hardware log builder: `308 passed`
- current `2026-05-19` result after Stage 4 A/B analyzer: `315 passed`
- current `2026-05-19` result after hardware binding validator: `321 passed`
- current `2026-05-19` result after hardware release gate: `326 passed`
- current `2026-05-19` result after hardware release packager: `331 passed`

Weak-hardware fast profile:

- `python -m pytest -q -m "not slow and not hardware"`
- `scripts/run_fast_tests.ps1` or `scripts/run_fast_tests.sh`
- current `2026-05-19` result after train3 refresh implementation: `274 passed, 18 deselected`
- current `2026-05-19` result after hardware log builder: `290 passed, 18 deselected`
- current `2026-05-19` result after Stage 4 A/B analyzer: `297 passed, 18 deselected`
- current `2026-05-19` result after hardware binding validator: `303 passed, 18 deselected`
- current `2026-05-19` result after hardware release gate: `308 passed, 18 deselected`
- current `2026-05-19` result after hardware release packager: `313 passed, 18 deselected`

Slow research profile:

- `python -m pytest -q -m "slow and not hardware"`
- `scripts/run_slow_tests.ps1` or `scripts/run_slow_tests.sh`

AIR56 UNO Q targeted deploy regression:

- `python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py tests/test_air56_unoq_hardware_acceptance.py tests/test_air56_unoq_build_hardware_report.py tests/test_air56_unoq_analyze_stage4_ab.py tests/test_air56_unoq_validate_hw_binding.py tests/test_air56_unoq_stage0_loopback.py`
- current `2026-05-19` deploy smoke targeted result after hardware log builder: `94 passed`
- current `2026-05-19` deploy smoke targeted result after Stage 4 A/B analyzer: `101 passed`
- current `2026-05-19` deploy smoke targeted result after hardware binding validator: `107 passed`
- current `2026-05-19` deploy smoke targeted result after hardware release gate: `112 passed`
- current `2026-05-19` deploy smoke targeted result after hardware release packager: `117 passed`

AIR56 UNO Q combined repo-side deploy smoke:

- `python tools/run_air56_unoq_deploy_smoke.py`

AIR56 UNO Q host firmware static checks:

- `python tools/check_air56_unoq_firmware_static.py --mode mock`
- `python tools/check_air56_unoq_firmware_static.py --mode production-port`
- The production-port host smoke is not a substitute for the final real STM32U585 inverter/HAL build; it verifies the repo firmware path links without the mock adapter when the production `air56_foc_*` contract is supplied.

AIR56 UNO Q physical hardware acceptance validator:

- `python tools/air56_unoq_validate_hw_binding.py --manifest arduino/air56_unoq_ready/hardware_binding.filled.json`
- `python tools/air56_unoq_analyze_stage4_ab.py --foc-no-load-csv <foc_no_load.csv> --foc-load-step-csv <foc_load_step.csv> --ai-no-load-csv <ai_no_load.csv> --ai-load-step-csv <ai_load_step.csv> --max-current-rms-a <limit> --out-json arduino/air56_unoq_ready/hardware_logs_template/stage4_ab_summary.real.json`
- `python tools/air56_unoq_build_hardware_report.py --board-id <board> --operator <name> --stage0-json <stage0.json> --stage1-json <stage1.json> --stage2-json <stage2.json> --stage2-csv <stage2.csv> --stage3-json <stage3.json> --stage4-json <stage4.json> --out-json arduino/air56_unoq_ready/hardware_acceptance_report.filled.json`
- `python tools/air56_unoq_hardware_acceptance.py --report arduino/air56_unoq_ready/hardware_acceptance_report.filled.json`
- `python tools/air56_unoq_hardware_release_gate.py --binding-manifest arduino/air56_unoq_ready/hardware_binding.filled.json --hardware-report arduino/air56_unoq_ready/hardware_acceptance_report.filled.json --deploy-smoke-json .tmp_pytest/air56_unoq_deploy_smoke.json --coverage-json .tmp_pytest/coverage_air56_unoq_gate.json`
- `python tools/air56_unoq_package_hardware_release.py --package-tag <tag> --out-dir <release package dir> --binding-manifest arduino/air56_unoq_ready/hardware_binding.filled.json --hardware-report arduino/air56_unoq_ready/hardware_acceptance_report.filled.json --deploy-smoke-json .tmp_pytest/air56_unoq_deploy_smoke.json --coverage-json .tmp_pytest/coverage_air56_unoq_gate.json`
- required result before hardware-ready claim: `hardware_ready = true`
  and `release_ready = true`

AIR56 UNO Q production-critical coverage gate:

- `python tools/check_air56_unoq_coverage_gate.py`
- current thresholds:
  - total AIR56 deploy subset: `>=75%`
  - protocol, Stage 0 loopback, firmware static compile, deploy smoke runner, Stage 4 A/B analyzer, hardware binding validator, hardware report builder, hardware acceptance validator, hardware release gate, hardware release packager: `>=95%`
  - Linux bridge helper/runtime module floor: `>=75%`

## Guardrails

- Do not weaken acceptance thresholds to keep the package green.
- Do not remove the preserved AO2 diagnosis/tuning artifacts.
- Do not create a new root plan while this file is current.
- Do not treat temporary probe configs under `outputs/` as canonical live configs.
- Do not use a Step27 scan state as acceptance evidence unless its signature includes hashes for config, candidate, acceptance envelope, and checkpoint content.
- Do not treat `AIR56_UNOQ_USE_MOCK_HW` as a motor-connected production build.
- Do not call the AIR56 UNO Q hardware deployment complete before the staged bring-up protocol passes on physical hardware.
- Do not claim whole-repository 100% test coverage; enforce coverage on the production-critical AIR56 UNO Q subset and keep broad research validation under smoke/release checks.
- The canonical AO2 live path is:
  - [env_research_ao2_32_4_3kw.py](C:/mic_theory/config/env_research_ao2_32_4_3kw.py)
  - [step27_ao2_current_repro_candidate_20260519.json](C:/mic_theory/config/step27_ao2_current_repro_candidate_20260519.json)
  - [checkpoint_registry.json](C:/mic_theory/config/checkpoint_registry.json)

## Optional Backlog

Nothing below is required for project completion anymore.

1. Expand universal onboarding proofs from the current release slice to a broader AO2-specific productization path.

Completed on `2026-04-12` after main project closure:

- monolithic `tools/reproduce_ieee_step28.py` was split into path/command builders without behavior change
- `step27_air56_acceptance.json` now has a backward-compatible generic successor:
  - `step27_motor_acceptance.json`
  - packaging/checklist/freeze/summary tools accept both names
  - new packages emit the generic file while preserving the legacy alias
