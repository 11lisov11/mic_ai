# MIC_AI

Repository for the MIC/AI motor-control research stack, reproducibility pipelines, and IEEE/PGUPS publication artifacts.

## Current Release Scope

As of `2026-04-12`, the active release scope is:

- `AIR56`
- `AL31`

`AO2` is **not removed** from the repository.
It is kept as a research backlog and a preserved experimental branch so the team can return to it later without losing the accumulated work.

Operational rule:

- release and submission closure are currently evaluated on `AIR56 + AL31`
- `AO2` remains available in `config/`, `outputs/`, `paper/`, and the training/evaluation tools as a suspended research track
- do not delete or overwrite `AO2` artifacts when preparing 2-motor release candidates

## What Is In Scope Now

- strict post-restore release closure for `AIR56 + AL31`
- `Step27 -> Step28 -> verify` reproducibility for the 2-motor release slice
- publication/package artifacts for the 2-motor release slice
- onboarding and engineering cleanup that do not depend on `AO2` closure

## What Is Out Of Scope For The Current Release

- strict `AO2` closure under the old 3-motor requirement
- reopening the 2-motor release to chase `AO2` unless the scope is explicitly changed again

## AO2 Status

`AO2` is preserved as research groundwork.

Current state:

- there is a valid research trail showing that `seed 505 / start_stop` can be fixed by a dedicated policy
- that fix is not yet globally deployable as a single controller
- current simple runtime dispatch is not sufficient

This means:

- `AO2` is paused, not discarded
- later work can resume either through a richer policy family or a redesigned runtime dispatch

## Main Entry Points

- [step27_pipeline.py](C:/mic_theory/tools/step27_pipeline.py): benchmark and acceptance runs
- [reproduce_ieee_step28.py](C:/mic_theory/tools/reproduce_ieee_step28.py): end-to-end IEEE reproduce/package pipeline
- [train_any_motor_pipeline.py](C:/mic_theory/tools/train_any_motor_pipeline.py): universal onboarding pipeline
- [train_3motors_pipeline.py](C:/mic_theory/tools/train_3motors_pipeline.py): multi-motor training pipeline
- [PROJECT_MASTER_PLAN.md](C:/mic_theory/PROJECT_MASTER_PLAN.md): active master plan

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the active 2-motor strict reproduce flow:

```bash
python tools/reproduce_ieee_step28.py ^
  --motors air56,al31 ^
  --mic-mode ai ^
  --ai-control-mode ai_id_ref ^
  --strict-verify ^
  --package-tag 20260412_postrestore_ai_2motors_release
```

Latest strict-verified 2-motor package:

- [20260412_postrestore_ai_2motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release)
- verify artifact: [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release/VERIFY_SUBMISSION_CANDIDATE.json)

Latest onboarding proofs for the active 2-motor scope:

- passport-only green run:
  - [any_motor_onboarding_report.json](C:/mic_theory/outputs/train_any_motor_pipeline/eval_2motor_rawskip_al31_20260412/any_motor_onboarding_report.json)
- passport + identification green run:
  - [any_motor_onboarding_report.json](C:/mic_theory/outputs/train_any_motor_pipeline/eval_2motor_identskip_al31_20260412/any_motor_onboarding_report.json)

Run the underlying Step27 benchmark only:

```bash
python tools/step27_pipeline.py ^
  --motors air56,al31 ^
  --mic-mode ai ^
  --ai-control-mode ai_id_ref ^
  --seed-perturbation ^
  --seed-perturb-level 0.2 ^
  --out-dir outputs/step27_2motors_current
```

## Repository Structure

- `config/`: motor and environment configs
- `mic_ai/`: AI, metrics, training, runtime tools
- `tools/`: orchestration and reproducibility scripts
- `tests/`: regression and smoke tests
- `paper/`: publication and submission artifacts
- `outputs/`: experimental and reproduce artifacts
- `docs/`: documentation and archived planning materials

## Notes

- RL checkpoints are not fully stored in git history.
- `AO2` artifacts are intentionally retained even though the current release scope is 2 motors.
- The onboarding pipeline default benchmark scope is also `air56,al31`; add `ao2` explicitly only when resuming backlog research.
- The root plan in [PROJECT_MASTER_PLAN.md](C:/mic_theory/PROJECT_MASTER_PLAN.md) has priority over older archived plans.
