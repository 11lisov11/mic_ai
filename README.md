# MIC_AI

Repository for the MIC/AI motor-control research stack, reproducibility pipelines, and IEEE/PGUPS publication artifacts.

## Current Status

As of the `2026-05-19` reproducibility audit, the current-code `3-motor` Step27 baseline is green:

- `AIR56`
- `AL31`
- `AO2`

Historical strict-verified release package:

- [20260412_postrestore_ai_3motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release)
- verify artifact: [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)
- `verification_ok = true`

Current-code Step27 reproducibility baseline after the `2026-05-19` full training refresh:

- `AIR56`: `+1.072%` avg power saving, `+0.112%` eta gain, `0` err failures
- `AL31`: `+3.455%` avg power saving, `+0.003%` eta gain, `0` err failures
- `AO2`: `+0.512%` avg power saving, `+1.724%` eta gain, `0` err failures
- all three have `envelope_fail_count = 0`
- strict recheck output: [final_selected_strict_recheck_20260519](C:/mic_theory/outputs/train3_fullprog_20260519/final_selected_strict_recheck_20260519)
- tracked refresh manifest: [20260519_train3_refresh](C:/mic_theory/paper/ieee_2026/data/release/20260519_train3_refresh/research_refresh_manifest.json)

Hardware-productization status as of `2026-05-20`:

- `AIR56 UNO Q` is the first board deployment path.
- The split architecture is implemented as a deploy package: STM32U585 owns FOC/safety/fallback, QRB2210/Linux runs the AI `id_ref` decision layer.
- The repo now contains the firmware hardware-adapter contract, Linux bridge startup/fallback checks, and a Stage 0-4 log-to-report hardware acceptance builder.
- Physical board deployment is not complete until the real STM32U585 FOC/inverter layer implements the `air56_foc_*` adapter symbols and passes the staged bring-up protocol.
- A second practical deploy path is prepared for a commercial Delta MS300 `VFD4A8MS21ANSAA` drive: PC/QRB2210 -> isolated USB-RS485 -> MS300 -> AIR56.
- The Delta MS300 path is repo-side ready for safe Modbus RTU self-check, read-only telemetry, guarded frequency writes, guarded run/stop, CSV logging, and staged bring-up.
- With a commercial VFD, MIC/AI cannot directly command the researched `id_ref` actuator; the MS300 owns fast current/vector loops, so this path is a slow supervisory frequency/profile layer until deeper drive-side controls are proven.

New theory branch status as of `2026-05-25`:

- `Safe Neural Horizon PWM with Event-Triggered Twin Feedback` is implemented as a host-level research scaffold.
- It adds an alpha-beta induction-motor model, two-level inverter vector model, protected AI-PWM Safety Gateway, neural-horizon controller, neural twin/event-feedback scaffold, tests, MC=100 smoke, MC=500 host evidence, a tracked 31-scenario host matrix, and a host trace/FFT/THD-like evidence package.
- Machine-checkable theory status: `host_theory_scaffold_ready = true`, `publication_theory_complete = false`.
- Machine-checkable trace status: `trace_fft_thd_evidence_ready = true`, `publication_plots_fft_thd_ready = true` for host simulation evidence only.
- Machine-checkable twin status: `trained_domain_randomized_twin_ready = true` for theta-conditioned host evidence only; it is not a production sensorless identifier.
- Current novelty claim is deliberately limited and machine-checkable: a distinct host-simulated architecture exists; there is still no claim of tuned-baseline superiority, MCU/HIL readiness, or bench proof.
- First findings are recorded in [safe_neural_horizon_pwm_research.md](C:/mic_theory/docs/safe_neural_horizon_pwm_research.md), including fixed modeling/control, release-discipline, dead-time-path, and fallback/loss-accounting bugs.

Historical milestone kept for provenance:

- [20260412_postrestore_ai_2motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release)

## AO2 Resolution

`AO2` is no longer a suspended backlog branch.

The current reproducible closure path is:

- diagnose the physical mismatch of the old `AO2` runtime config
- rebuild `AO2` around a nameplate-first operating point
- add optional `field_weakening` support to FOC
- retune the live `AO2` config and keep the tuned AI actor
- retune the `AO2` supervisor/id_ref candidate after the `2026-05-19` stale-cache audit
- verify the result under current strict `Step27` envelope checks

The diagnostic trail is intentionally kept in the repository:

- [env_backlog_ao2_nameplate_first.py](C:/mic_theory/config/env_backlog_ao2_nameplate_first.py)
- [env_backlog_ao2_nameplate_foc_tuned.py](C:/mic_theory/config/env_backlog_ao2_nameplate_foc_tuned.py)
- [diagnose_motor_nominal_consistency.py](C:/mic_theory/tools/diagnose_motor_nominal_consistency.py)
- current AO2 candidate: [step27_ao2_current_repro_candidate_20260519.json](C:/mic_theory/config/step27_ao2_current_repro_candidate_20260519.json)
- current AO2 strict scan: [ao2_current_repro_strict_scan_20260519](C:/mic_theory/outputs/ao2_current_repro_strict_scan_20260519)
- old `fw_c` scan is preserved as provenance, but it is no longer the canonical current-code acceptance source because the `2026-05-19` audit found stale resume-cache risk in Step27 scans.

## 2026-05-19 Train3 Refresh

The latest full training pass is captured as a tracked refresh release:

- manifest: [research_refresh_manifest.json](C:/mic_theory/paper/ieee_2026/data/release/20260519_train3_refresh/research_refresh_manifest.json)
- `AIR56`: keep accepted canonical baseline
- `AL31`: promote the fine-tuned `actor_ep018.pth` result through `best_actor_step27_train3.pth`
- `AO2`: keep accepted nameplate-first canonical baseline
- `research_refresh_complete = true`
- `hardware_deploy_complete = false`

The promoted `AL31` checkpoint is under ignored `outputs/` by design. The tracked manifest records the path, SHA-256, metrics, and reproduce commands.

## Main Entry Points

- [step27_pipeline.py](C:/mic_theory/tools/step27_pipeline.py): benchmark and acceptance runs
- [reproduce_ieee_step28.py](C:/mic_theory/tools/reproduce_ieee_step28.py): end-to-end IEEE reproduce/package pipeline
- [train_any_motor_pipeline.py](C:/mic_theory/tools/train_any_motor_pipeline.py): universal onboarding pipeline
- [train_3motors_pipeline.py](C:/mic_theory/tools/train_3motors_pipeline.py): multi-motor training pipeline
- [air56_unoq_bridge.py](C:/mic_theory/tools/air56_unoq_bridge.py): QRB2210 Linux bridge for AIR56 UNO Q
- [air56_unoq_stage0_loopback.py](C:/mic_theory/tools/air56_unoq_stage0_loopback.py): Stage 0 protocol self-test
- [air56_unoq_analyze_stage4_ab.py](C:/mic_theory/tools/air56_unoq_analyze_stage4_ab.py): analyzes physical AIR56 Stage 4 FOC vs MIC/AI A/B CSV logs
- [air56_unoq_build_hardware_report.py](C:/mic_theory/tools/air56_unoq_build_hardware_report.py): builds a Stage 0-4 acceptance report from real board logs
- [air56_unoq_validate_hw_binding.py](C:/mic_theory/tools/air56_unoq_validate_hw_binding.py): validates STM32U585 pinout/scaling/fault binding and production `air56_foc_*` adapter source
- [air56_unoq_hardware_acceptance.py](C:/mic_theory/tools/air56_unoq_hardware_acceptance.py): validator for real Stage 0-4 hardware evidence
- [air56_unoq_hardware_release_gate.py](C:/mic_theory/tools/air56_unoq_hardware_release_gate.py): final AIR56 hardware release gate aggregator
- [air56_unoq_package_hardware_release.py](C:/mic_theory/tools/air56_unoq_package_hardware_release.py): packages final AIR56 hardware release evidence with SHA-256 hashes
- [run_air56_unoq_deploy_smoke.py](C:/mic_theory/tools/run_air56_unoq_deploy_smoke.py): one-command AIR56 UNO Q repo-side smoke
- [air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready): AIR56 UNO Q split deploy package
- [air56_unoq_bringup.md](C:/mic_theory/docs/air56_unoq_bringup.md): physical bring-up protocol
- [delta_ms300_modbus.py](C:/mic_theory/tools/delta_ms300_modbus.py): Delta MS300 Modbus RTU protocol, safety gates, and CLI
- [delta_ms300_modbus_bridge.py](C:/mic_theory/tools/delta_ms300_modbus_bridge.py): Delta MS300 bridge wrapper
- [run_delta_ms300_deploy_smoke.py](C:/mic_theory/tools/run_delta_ms300_deploy_smoke.py): one-command Delta MS300 repo-side smoke
- [vfd_delta_ms300_air56.json](C:/mic_theory/config/vfd_delta_ms300_air56.json): safe default AIR56 Delta MS300 config
- [delta_ms300_air56_ready](C:/mic_theory/vfd/delta_ms300_air56_ready): Delta MS300 deploy package
- [delta_ms300_air56_bringup.md](C:/mic_theory/docs/delta_ms300_air56_bringup.md): Delta MS300 physical bring-up protocol
- [induction_motor_alpha_beta.py](C:/mic_theory/models/induction_motor_alpha_beta.py): alpha-beta induction motor model for the new Safety Neural Horizon PWM research track
- [two_level_inverter.py](C:/mic_theory/models/two_level_inverter.py): key-level two-level inverter vector model
- [ai_pwm_gateway.py](C:/mic_theory/safety/ai_pwm_gateway.py): protected AI-PWM Safety Gateway and no-shoot-through waveform helpers
- [deadbeat_current_baseline.py](C:/mic_theory/control/deadbeat_current_baseline.py): host deadbeat predictive current-control baseline with Safety Gateway protection
- [dtc_baseline.py](C:/mic_theory/control/dtc_baseline.py): host DTC hysteresis comparison baseline over legal inverter vectors with Safety Gateway protection
- [dtc_svm_baseline.py](C:/mic_theory/control/dtc_svm_baseline.py): host DTC-SVM comparison baseline with torque/flux voltage synthesis and Safety Gateway protection
- [fcs_mpc_baseline.py](C:/mic_theory/control/fcs_mpc_baseline.py): host one-step FCS-MPC comparison baseline over legal inverter vectors with Safety Gateway protection
- [foc_svm_key_baseline.py](C:/mic_theory/control/foc_svm_key_baseline.py): host key-level FOC-SVM comparison baseline with PI speed/current loops and Safety Gateway protection
- [protected_ai_pwm_h1_baseline.py](C:/mic_theory/control/protected_ai_pwm_h1_baseline.py): host prior protected AI-PWM H1 baseline used to compare the new horizon variants against the previous one-step protected architecture
- [sensorless_adaptive_foc_baseline.py](C:/mic_theory/control/sensorless_adaptive_foc_baseline.py): host sensorless/adaptive FOC baseline with MRAS-like speed observer, Rs adaptation, and Safety Gateway protection
- [safe_neural_horizon_pwm.py](C:/mic_theory/control/safe_neural_horizon_pwm.py): event-triggered neural-horizon AI-PWM controller scaffold
- [run_safe_neural_horizon_pwm_study.py](C:/mic_theory/tools/run_safe_neural_horizon_pwm_study.py): quick host-level MC smoke for the new research track
- [build_safe_neural_horizon_pwm_report.py](C:/mic_theory/tools/build_safe_neural_horizon_pwm_report.py): builds a markdown report from Safe Neural Horizon PWM JSON results
- [build_safe_neural_horizon_pwm_figures.py](C:/mic_theory/tools/build_safe_neural_horizon_pwm_figures.py): builds aggregate SVG/CSV figures from Safe Neural Horizon PWM JSON results
- [check_safe_neural_horizon_pwm_release.py](C:/mic_theory/tools/check_safe_neural_horizon_pwm_release.py): validates the host-release evidence and SHA-256 manifest
- [check_safe_neural_horizon_pwm_novelty.py](C:/mic_theory/tools/check_safe_neural_horizon_pwm_novelty.py): audits the allowed host-level novelty claim and explicitly rejects overclaims
- [check_safe_neural_horizon_pwm_theory.py](C:/mic_theory/tools/check_safe_neural_horizon_pwm_theory.py): audits whether the new theory branch is host-scaffold-ready and blocks publication-grade overclaims
- [package_safe_neural_horizon_pwm_release.py](C:/mic_theory/tools/package_safe_neural_horizon_pwm_release.py): packages Safe Neural Horizon PWM host-simulation release evidence
- [safe_neural_horizon_pwm_research.md](C:/mic_theory/docs/safe_neural_horizon_pwm_research.md): current theory report, limitations, and next work
- [20260522_host_release](C:/mic_theory/paper/safe_neural_horizon_pwm_2026/20260522_host_release): tracked Safe Neural Horizon PWM host-level release package
- [PROJECT_MASTER_PLAN.md](C:/mic_theory/PROJECT_MASTER_PLAN.md): active root status and guardrails

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the strict 3-motor reproduce flow:

```bash
python tools/reproduce_ieee_step28.py ^
  --motors air56,al31,ao2 ^
  --mic-mode ai ^
  --ai-control-mode ai_id_ref ^
  --strict-verify ^
  --package-tag 20260412_postrestore_ai_3motors_release
```

Run the underlying strict Step27 benchmark only:

```bash
python tools/step27_pipeline.py ^
  --motors air56,al31,ao2 ^
  --mic-mode ai ^
  --ai-control-mode ai_id_ref ^
  --seed-perturbation ^
  --seed-perturb-level 0.2 ^
  --out-dir outputs/step27_3motors_current
```

Run the Delta MS300 repo-side smoke before connecting hardware:

```bash
python tools/run_delta_ms300_deploy_smoke.py
```

Run the new Safe Neural Horizon PWM host-level smoke:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 100 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc100.json
```

Run the host-level scenario/ablation/Pareto matrix for the new research track:

```bash
python tools/run_safe_neural_horizon_pwm_study.py --matrix --mc 3 --steps 60 --out-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json
python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 100 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc100.json
python tools/run_safe_neural_horizon_pwm_study.py --quick --mc 500 --steps 120 --out-json .tmp_pytest/safe_neural_horizon_pwm_study_mc500.json
python tools/build_safe_neural_horizon_pwm_trace_evidence.py --steps 512 --out-dir .tmp_pytest/safe_neural_horizon_pwm_trace_evidence
python tools/build_safe_neural_horizon_pwm_twin_evidence.py --out-dir .tmp_pytest/safe_neural_horizon_pwm_twin_evidence
python tools/package_safe_neural_horizon_pwm_release.py --input-json .tmp_pytest/safe_neural_horizon_pwm_full_host_matrix_mc3.json --out-dir paper/safe_neural_horizon_pwm_2026/20260522_host_release --tag 20260522_safe_neural_horizon_pwm_host_release --trace-dir .tmp_pytest/safe_neural_horizon_pwm_trace_evidence --twin-dir .tmp_pytest/safe_neural_horizon_pwm_twin_evidence --mc500-json .tmp_pytest/safe_neural_horizon_pwm_study_mc500.json
python tools/check_safe_neural_horizon_pwm_release.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_novelty.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
python tools/check_safe_neural_horizon_pwm_theory.py --input paper/safe_neural_horizon_pwm_2026/20260522_host_release --strict
```

Run a read-only Delta MS300 check after wiring an isolated USB-RS485 adapter and
editing `config/vfd_delta_ms300_air56.json` for the real COM port:

```bash
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json self-check
python tools/delta_ms300_modbus_bridge.py --config config/vfd_delta_ms300_air56.json read-once
```

## Validation Snapshot

- historical strict 3-motor `Step28` verify: green
- current-code strict Step27 full-training refresh: green for `AIR56`, `AL31`, `AO2`
- `train_3motors_pipeline.py --step27-select` full refresh: `3/3` runs passed; `AIR56`/`AO2` kept canonical baselines, `AL31` promoted the new fine-tuned checkpoint
- Step27 scan resume-state now hashes config, candidate, acceptance envelope, and checkpoint content to prevent stale acceptance reuse.
- `AO2` live config now uses `ao2_current_repro_rand017`
  - [motor_tuning_acceptance_summary.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_tuning_acceptance_summary.json)
- latest focused regression after AO2 FOC/hybrid fixes:
  - `python -m pytest -q tests/test_step27_report_markdown.py tests/test_step27_hybrid_trigger.py tests/test_vector_foc_field_weakening.py tests/test_scan_step27_checkpoints.py tests/test_train_ai_id_ref_external_step27.py tests/test_diagnose_motor_nominal_consistency.py`
  - prior audit result: `60 passed`
- latest focused regression after the `2026-05-19` training/Step27 fixes:
  - `python -m pytest -q tests/test_scan_step27_checkpoints.py tests/test_tune_motor_step27_candidates.py tests/test_train_3motors_pipeline_smoke.py tests/test_train_3motors_pipeline_joint_and_finetune_smoke.py tests/test_train_3motors_pipeline_resume_eval_first_smoke.py`
  - `34 passed`
- latest full training refresh:
  - joint domain-randomized seed `101`: complete
  - fine-tune per motor seed `101`: complete
  - selected strict recheck: `3/3` pass
- AIR56 UNO Q focused deploy regression:
  - `python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py tests/test_air56_unoq_hardware_acceptance.py tests/test_air56_unoq_build_hardware_report.py tests/test_air56_unoq_analyze_stage4_ab.py tests/test_air56_unoq_validate_hw_binding.py tests/test_air56_unoq_stage0_loopback.py`
- AIR56 UNO Q firmware static compile/link smoke:
  - `python tools/check_air56_unoq_firmware_static.py --mode mock`
  - `python tools/check_air56_unoq_firmware_static.py --mode production-port`
- AIR56 UNO Q STM32U585 hardware binding validator:
  - `python tools/air56_unoq_validate_hw_binding.py --manifest arduino/air56_unoq_ready/hardware_binding.filled.json`
  - required before physical Stage 1 claim: `hardware_binding_ready = true`
- AIR56 UNO Q one-command repo-side deploy smoke:
  - `python tools/run_air56_unoq_deploy_smoke.py`
  - current `2026-05-19` result after hardware log builder: `passed = true`, targeted pytest `94 passed`
  - current `2026-05-19` result after Stage 4 A/B analyzer: `passed = true`, targeted pytest `101 passed`
  - current `2026-05-19` result after hardware binding validator: `passed = true`, targeted pytest `107 passed`
  - current `2026-05-19` result after hardware release gate: `passed = true`, targeted pytest `112 passed`
  - current `2026-05-19` result after hardware release packager: `passed = true`, targeted pytest `117 passed`
- AIR56 UNO Q hardware report builder from real Stage 0-4 logs:
  - `python tools/air56_unoq_analyze_stage4_ab.py --foc-no-load-csv <foc_no_load.csv> --foc-load-step-csv <foc_load_step.csv> --ai-no-load-csv <ai_no_load.csv> --ai-load-step-csv <ai_load_step.csv> --max-current-rms-a <limit> --out-json <stage4_ab_summary.json>`
  - `python tools/air56_unoq_build_hardware_report.py --board-id <board> --operator <name> --stage0-json <stage0.json> --stage1-json <stage1.json> --stage2-json <stage2.json> --stage2-csv <stage2.csv> --stage3-json <stage3.json> --stage4-json <stage4.json> --out-json <hardware_report.json>`
- AIR56 UNO Q physical hardware acceptance validator:
  - `python tools/air56_unoq_hardware_acceptance.py --report arduino/air56_unoq_ready/hardware_acceptance_report.filled.json`
  - required result before hardware-ready claim: `hardware_ready = true`
- AIR56 UNO Q final hardware release gate:
  - `python tools/air56_unoq_hardware_release_gate.py --binding-manifest <hardware_binding.filled.json> --hardware-report <hardware_acceptance_report.filled.json> --deploy-smoke-json <deploy_smoke.json> --coverage-json <coverage_air56_unoq_gate.json>`
  - required result before release-ready claim: `release_ready = true`
- AIR56 UNO Q final hardware release package:
  - `python tools/air56_unoq_package_hardware_release.py --package-tag <tag> --out-dir <release_dir> --binding-manifest <hardware_binding.filled.json> --hardware-report <hardware_acceptance_report.filled.json> --deploy-smoke-json <deploy_smoke.json> --coverage-json <coverage_air56_unoq_gate.json>`
  - writes `hardware_release_manifest.json` with SHA-256 hashes for every evidence file
- AIR56 UNO Q production-critical coverage gate:
  - `python tools/check_air56_unoq_coverage_gate.py`
  - current gate: total `>=75%`, protocol/loopback/static/deploy-smoke/stage4-analyzer/hardware-binding/hardware-report/hardware-acceptance/release-gate/release-packager `>=95%`, bridge helper/runtime floor `>=75%`
  - current `2026-05-19` result after hardware log builder: `passed = true`, total AIR56 deploy subset coverage `89.48%`, `tools/air56_unoq_build_hardware_report.py = 100.00%`
  - current `2026-05-19` result after Stage 4 A/B analyzer: `passed = true`, total AIR56 deploy subset coverage `90.70%`, `tools/air56_unoq_analyze_stage4_ab.py = 100.00%`
  - current `2026-05-19` result after hardware binding validator: `passed = true`, total AIR56 deploy subset coverage `91.57%`, `tools/air56_unoq_validate_hw_binding.py = 100.00%`
  - current `2026-05-19` result after hardware release gate: `passed = true`, total AIR56 deploy subset coverage `92.06%`, `tools/air56_unoq_hardware_release_gate.py = 100.00%`
  - current `2026-05-19` result after hardware release packager: `passed = true`, total AIR56 deploy subset coverage `92.42%`, `tools/air56_unoq_package_hardware_release.py = 100.00%`
- Delta MS300 AIR56 repo-side deploy smoke:
  - `python tools/run_delta_ms300_deploy_smoke.py`
  - current `2026-05-20` result: `passed = true`, targeted pytest `20 passed`
- Delta MS300 AIR56 targeted regression:
  - `python -m pytest -q tests/test_delta_ms300_modbus.py`
  - current `2026-05-20` result: `20 passed`
- weak-hardware fast profile:
  - `python -m pytest -q -m "not slow and not hardware"`
  - `scripts/run_fast_tests.ps1` or `scripts/run_fast_tests.sh`
  - current `2026-05-19` result after hardware log builder: `290 passed, 18 deselected`
  - current `2026-05-19` result after Stage 4 A/B analyzer: `297 passed, 18 deselected`
  - current `2026-05-19` result after hardware binding validator: `303 passed, 18 deselected`
  - current `2026-05-19` result after hardware release gate: `308 passed, 18 deselected`
  - current `2026-05-19` result after hardware release packager: `313 passed, 18 deselected`
  - current `2026-05-20` result after Delta MS300 deploy package: `333 passed, 18 deselected`
- slow research profile:
  - `python -m pytest -q -m "slow and not hardware"`
  - `scripts/run_slow_tests.ps1` or `scripts/run_slow_tests.sh`
- full repository regression after train3 refresh implementation:
  - `python -m pytest -q`
  - current `2026-05-19` result after hardware log builder: `308 passed`
  - current `2026-05-19` result after Stage 4 A/B analyzer: `315 passed`
  - current `2026-05-19` result after hardware binding validator: `321 passed`
  - current `2026-05-19` result after hardware release gate: `326 passed`
  - current `2026-05-19` result after hardware release packager: `331 passed`
  - current `2026-05-20` result after Delta MS300 deploy package: `351 passed`

## Repository Structure

- `config/`: motor and environment configs
- `control/`: low-level controllers including FOC
- `mic_ai/`: AI, metrics, training, runtime tools
- `tools/`: orchestration and reproducibility scripts
- `tests/`: regression and smoke tests
- `vfd/`: commercial VFD deploy packages such as Delta MS300 AIR56
- `paper/`: publication and submission artifacts
- `outputs/`: experimental and reproduce artifacts
- `docs/`: documentation and archived planning materials

## Notes

- RL checkpoints are not fully stored in git history.
- The canonical `AO2` live config is now [env_research_ao2_32_4_3kw.py](C:/mic_theory/config/env_research_ao2_32_4_3kw.py).
- The canonical current-code `AO2` candidate is [step27_ao2_current_repro_candidate_20260519.json](C:/mic_theory/config/step27_ao2_current_repro_candidate_20260519.json).
- The canonical checkpoint registry is [checkpoint_registry.json](C:/mic_theory/config/checkpoint_registry.json).
- The root plan in [PROJECT_MASTER_PLAN.md](C:/mic_theory/PROJECT_MASTER_PLAN.md) has priority over archived plans.
- `AIR56` deploy package for `UNO Q` is available in [arduino/air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready). It is a split hardware-productization package, not proof that a motor-connected STM32U585 build has already passed physical acceptance.
- The hardware report builder is fail-safe: missing or incomplete board logs produce `hardware_ready=false`, not a simulated pass.
- The Stage 4 A/B analyzer is fail-safe: it only passes when physical MIC/AI logs do not regress power, tracking, guard failures, current/thermal limits, or fallback stability against FOC logs.
- The hardware binding validator is fail-safe: template manifests, mock adapter source, constant-return stubs, and undocumented `air56_foc_*` mappings produce `hardware_binding_ready=false`.
- The hardware release gate is fail-safe: it only passes when binding, physical acceptance, deploy smoke, and coverage evidence are all green.
- The hardware release packager is fail-safe by default: it still writes evidence for diagnosis, but returns nonzero unless `release_ready=true` or `--allow-not-ready` is explicitly used.
- `Delta MS300` support is fail-safe by default: checked-in config does not allow writes or run commands until both the JSON safety flags and CLI arming flags are explicitly enabled.
- The Delta MS300 path is a commercial-VFD supervisory path, not a direct replacement for the STM32U585 `id_ref` actuator used in the research release.
- Whole-repository coverage is not expected to be 100% because this repo contains many research CLI and long-running reproduction scripts. Coverage gating is enforced on the production-critical AIR56 UNO Q deploy subset instead.
