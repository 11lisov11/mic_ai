# MIC_AI

Repository for the MIC/AI motor-control research stack, reproducibility pipelines, and IEEE/PGUPS publication artifacts.

## Current Status

As of `2026-04-12`, the full `3-motor` project is closed:

- `AIR56`
- `AL31`
- `AO2`

Canonical strict-verified release:

- [20260412_postrestore_ai_3motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release)
- verify artifact: [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)
- `verification_ok = true`

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
- Ready `AIR56` deploy package for `UNO Q` is available in [arduino/air56_unoq_ready](C:/mic_theory/arduino/air56_unoq_ready).

