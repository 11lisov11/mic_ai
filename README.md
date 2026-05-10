# MIC_AI

Repository for the MIC/AI motor-control research stack, reproducibility pipelines, and IEEE/PGUPS publication artifacts.

## Current Status

As of `2026-04-12`, the full `3-motor` research/release project is closed:

- `AIR56`
- `AL31`
- `AO2`

Canonical strict-verified release:

- [20260412_postrestore_ai_3motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release)
- verify artifact: [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)
- `verification_ok = true`

Hardware-productization status as of `2026-05-05`:

- `AIR56 UNO Q` is the first board deployment path.
- The split architecture is implemented as a deploy package: STM32U585 owns FOC/safety/fallback, QRB2210/Linux runs the AI `id_ref` decision layer.
- The repo now contains the firmware hardware-adapter contract and Linux bridge startup/fallback checks.
- Physical board deployment is not complete until the real STM32U585 FOC/inverter layer implements the `air56_foc_*` adapter symbols and passes the staged bring-up protocol.

Historical milestone kept for provenance:

- [20260412_postrestore_ai_2motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release)

## AO2 Resolution

`AO2` is no longer a suspended backlog branch.

The final closure path was:

- diagnose the physical mismatch of the old `AO2` runtime config
- rebuild `AO2` around a nameplate-first operating point
- add optional `field_weakening` support to FOC
- retune the live `AO2` config and keep the tuned AI actor
- verify the result under strict `Step27/Step28` `p0.2`

The diagnostic trail is intentionally kept in the repository:

- [env_backlog_ao2_nameplate_first.py](C:/mic_theory/config/env_backlog_ao2_nameplate_first.py)
- [env_backlog_ao2_nameplate_foc_tuned.py](C:/mic_theory/config/env_backlog_ao2_nameplate_foc_tuned.py)
- [diagnose_motor_nominal_consistency.py](C:/mic_theory/tools/diagnose_motor_nominal_consistency.py)
- [ao2 fw strict pass](C:/mic_theory/outputs/ao2_fw_grid_20260412af/fw_c/ao2_checkpoint_scan_summary.json)

## Main Entry Points

- [step27_pipeline.py](C:/mic_theory/tools/step27_pipeline.py): benchmark and acceptance runs
- [reproduce_ieee_step28.py](C:/mic_theory/tools/reproduce_ieee_step28.py): end-to-end IEEE reproduce/package pipeline
- [train_any_motor_pipeline.py](C:/mic_theory/tools/train_any_motor_pipeline.py): universal onboarding pipeline
- [train_3motors_pipeline.py](C:/mic_theory/tools/train_3motors_pipeline.py): multi-motor training pipeline
- [air56_unoq_bridge.py](C:/mic_theory/tools/air56_unoq_bridge.py): QRB2210 Linux bridge for AIR56 UNO Q
- [air56_unoq_stage0_loopback.py](C:/mic_theory/tools/air56_unoq_stage0_loopback.py): Stage 0 protocol self-test
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

- strict 3-motor `Step28` verify: green
- `AO2` motor acceptance in the release package: green
  - [motor_tuning_acceptance_summary.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_tuning_acceptance_summary.json)
- latest focused regression after AO2 FOC/hybrid fixes:
  - `python -m pytest -q tests/test_step27_report_markdown.py tests/test_step27_hybrid_trigger.py tests/test_vector_foc_field_weakening.py tests/test_scan_step27_checkpoints.py tests/test_train_ai_id_ref_external_step27.py tests/test_diagnose_motor_nominal_consistency.py`
  - `60 passed`
- AIR56 UNO Q focused deploy regression:
  - `python -m pytest -q tests/test_uno_q_protocol.py tests/test_uno_q_bridge.py tests/test_air56_unoq_bridge.py tests/test_air56_unoq_deploy_package.py`
- AIR56 UNO Q firmware static compile smoke:
  - `python tools/check_air56_unoq_firmware_static.py`
- AIR56 UNO Q one-command repo-side deploy smoke:
  - `python tools/run_air56_unoq_deploy_smoke.py`
- AIR56 UNO Q production-critical coverage gate:
  - `python tools/check_air56_unoq_coverage_gate.py`
  - current gate: total `>=75%`, protocol/loopback/static/deploy-smoke `>=95%`, bridge helper/runtime floor `>=75%`
- weak-hardware fast profile:
  - `python -m pytest -q -m "not slow and not hardware"`

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
- The canonical checkpoint registry is [checkpoint_registry.json](C:/mic_theory/config/checkpoint_registry.json).
- The root plan in [PROJECT_MASTER_PLAN.md](C:/mic_theory/PROJECT_MASTER_PLAN.md) has priority over archived plans.
- `AIR56` deploy package for `UNO Q` is available in [arduino/air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready). It is a split hardware-productization package, not proof that a motor-connected STM32U585 build has already passed physical acceptance.
- Whole-repository coverage is not expected to be 100% because this repo contains many research CLI and long-running reproduction scripts. Coverage gating is enforced on the production-critical AIR56 UNO Q deploy subset instead.
