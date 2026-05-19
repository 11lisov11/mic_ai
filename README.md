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

Hardware-productization status as of `2026-05-19`:

- `AIR56 UNO Q` is the first board deployment path.
- The split architecture is implemented as a deploy package: STM32U585 owns FOC/safety/fallback, QRB2210/Linux runs the AI `id_ref` decision layer.
- The repo now contains the firmware hardware-adapter contract, Linux bridge startup/fallback checks, and a Stage 0-4 log-to-report hardware acceptance builder.
- Physical board deployment is not complete until the real STM32U585 FOC/inverter layer implements the `air56_foc_*` adapter symbols and passes the staged bring-up protocol.

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
- [air56_unoq_build_hardware_report.py](C:/mic_theory/tools/air56_unoq_build_hardware_report.py): builds a Stage 0-4 acceptance report from real board logs
- [air56_unoq_hardware_acceptance.py](C:/mic_theory/tools/air56_unoq_hardware_acceptance.py): validator for real Stage 0-4 hardware evidence
- [run_air56_unoq_deploy_smoke.py](C:/mic_theory/tools/run_air56_unoq_deploy_smoke.py): one-command AIR56 UNO Q repo-side smoke
- [air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready): AIR56 UNO Q split deploy package
- [air56_unoq_bringup.md](C:/mic_theory/docs/air56_unoq_bringup.md): physical bring-up protocol
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
  - `python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py tests/test_air56_unoq_hardware_acceptance.py tests/test_air56_unoq_build_hardware_report.py tests/test_air56_unoq_stage0_loopback.py`
- AIR56 UNO Q firmware static compile/link smoke:
  - `python tools/check_air56_unoq_firmware_static.py --mode mock`
  - `python tools/check_air56_unoq_firmware_static.py --mode production-port`
- AIR56 UNO Q one-command repo-side deploy smoke:
  - `python tools/run_air56_unoq_deploy_smoke.py`
  - current `2026-05-19` result after hardware log builder: `passed = true`, targeted pytest `94 passed`
- AIR56 UNO Q hardware report builder from real Stage 0-4 logs:
  - `python tools/air56_unoq_build_hardware_report.py --board-id <board> --operator <name> --stage0-json <stage0.json> --stage1-json <stage1.json> --stage2-json <stage2.json> --stage2-csv <stage2.csv> --stage3-json <stage3.json> --stage4-json <stage4.json> --out-json <hardware_report.json>`
- AIR56 UNO Q physical hardware acceptance validator:
  - `python tools/air56_unoq_hardware_acceptance.py --report arduino/air56_unoq_ready/hardware_acceptance_report.filled.json`
  - required result before hardware-ready claim: `hardware_ready = true`
- AIR56 UNO Q production-critical coverage gate:
  - `python tools/check_air56_unoq_coverage_gate.py`
  - current gate: total `>=75%`, protocol/loopback/static/deploy-smoke/hardware-report/hardware-acceptance `>=95%`, bridge helper/runtime floor `>=75%`
  - current `2026-05-19` result after hardware log builder: `passed = true`, total AIR56 deploy subset coverage `89.48%`, `tools/air56_unoq_build_hardware_report.py = 100.00%`
- weak-hardware fast profile:
  - `python -m pytest -q -m "not slow and not hardware"`
  - `scripts/run_fast_tests.ps1` or `scripts/run_fast_tests.sh`
  - current `2026-05-19` result after hardware log builder: `290 passed, 18 deselected`
- slow research profile:
  - `python -m pytest -q -m "slow and not hardware"`
  - `scripts/run_slow_tests.ps1` or `scripts/run_slow_tests.sh`
- full repository regression after train3 refresh implementation:
  - `python -m pytest -q`
  - current `2026-05-19` result after hardware log builder: `308 passed`

## Repository Structure

- `config/`: motor and environment configs
- `control/`: low-level controllers including FOC
- `mic_ai/`: AI, metrics, training, runtime tools
- `tools/`: orchestration and reproducibility scripts
- `tests/`: regression and smoke tests
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
- Whole-repository coverage is not expected to be 100% because this repo contains many research CLI and long-running reproduction scripts. Coverage gating is enforced on the production-critical AIR56 UNO Q deploy subset instead.
