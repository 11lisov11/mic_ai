# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-03-16`
Repository: `c:\mic_theory`

## Canonical source
- This file is the only active master plan allowed in the repository root.
- Historical root plans and root execution logs are archived under `docs/plan_archive/`.
- As of this refresh, the previous root files were moved to:
  - `docs/plan_archive/2026-03-16_plan_refresh/PROJECT_MASTER_PLAN_20260311_snapshot.md`
  - `docs/plan_archive/2026-03-16_plan_refresh/PROJECT_MASTER_EXECUTION_LOG_20260303_cycle2.md`

## Current factual baseline
- Test baseline is green:
  - `pytest -q` -> `117 passed` on `2026-03-16`.
- Stable release path is already closed for the frozen tag:
  - `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/VERIFY_SUBMISSION_CANDIDATE.json`
  - `checklist_ready_for_submission=true`
  - `verification_ok=true`
- Universal onboarding is implemented and usable:
  - `tools/train_any_motor_pipeline.py`
  - `docs/any_motor_onboarding.md`
  - `tests/test_train_any_motor_pipeline_smoke.py`
- Universal onboarding correctness gate is closed for the demo flow when revalidating from an existing checkpoint:
  - `outputs/train_any_motor_pipeline/eval_demo_any_reval_gate_default2/any_motor_onboarding_report.json`
  - `all_ok=true`
  - `pass_count=3/3`
- Universal onboarding energy gate is not closed for the demo flow:
  - `outputs/train_any_motor_pipeline/eval_demo_any_plan_v2/any_motor_onboarding_report.json`
  - `all_ok=false`
  - `pass_count=2/3`
  - bottleneck: `ao2 power_saving_pct_mean = -1.18827547060954`
- Post-restore research branch is not closed:
  - `paper/ieee_2026/data/step28/20260308_postrestore_ai/VERIFY_SUBMISSION_CANDIDATE.json`
  - `checklist_ready_for_submission=false`
  - `verification_ok=false`
- Post-restore envelope gate is not closed:
  - `outputs/step27_baseline_20260308_postrestore_s3_fixao2/acceptance_envelopes/acceptance_envelope_summary.json`
  - `all_rows_pass=false`
- Low-compute W1 sweep was executed on `2026-03-16`:
  - `tools/tune_motor_step27.py` now supports targeted candidate input via `--candidate-json`
  - `tests/test_tune_motor_step27_candidates.py` protects that loader
  - focused AIR56/AL31/AO2 config-only sweeps were run on current checkpoints
  - short AO2 warm-start pilots were run from the existing checkpoint
  - result: no cheap closure; the red branch did not become green without checkpoint-level retraining/finetuning

## What is still not finished to 100%
1. A new post-restore frozen release has not been produced.
2. The universal any-motor algorithm is not yet closed on the energy-efficiency criterion.
3. The onboarding flow is not yet proven on a real identification-first scenario with hardware data.
4. The main orchestration layer still contains oversized monolithic scripts and duplicated responsibilities.
5. Documentation and repository hygiene are not fully cleaned:
   - `README.md` currently has a broken encoding view in the shell and needs normalization to UTF-8.
   - Root planning hygiene needed cleanup and must remain enforced.
6. Test coverage is still shallow for the newest onboarding modes:
   - no dedicated smoke/regression test for `--skip-training --init-checkpoint`
   - no test for benchmark search ranking and gate success/failure modes
   - no regression that protects doc encoding / root plan hygiene

## Definition of 100% done
The project is considered finished only when all items below are true.

- A post-restore research tag exists with:
  - `ready_for_submission=true`
  - `checklist_ready_for_submission=true`
  - `verification_ok=true`
- The universal onboarding track has:
  - correctness gate green on the benchmark trio (`air56`, `al31`, `ao2`)
  - energy gate green on the designated verification set
  - documented passport-only flow
  - documented passport + identification flow
  - reproducible report artifacts for both
- Refactor work is complete for active orchestration scripts:
  - `tools/step27_pipeline.py`
  - `tools/train_any_motor_pipeline.py`
  - `tools/train_3motors_pipeline.py`
  - `tools/robust_motor_hardening.py`
- Documentation is clean and current:
  - `README.md` is readable in UTF-8
  - runbooks match the real entrypoints and gate semantics
  - root contains only one active plan
- `pytest -q` remains green after all cleanup.

## Do not redo
- Do not re-open the frozen release `20260303_ai_config_locked_nodrift` unless a bug is found in its artifacts.
- Do not rewrite historical release bundles under `paper/ieee_2026/submission_bundle/` just for cleanup.
- Do not weaken acceptance thresholds to force a green result.
- Do not create new dated plan files in root.

## Active workstreams

### W0. Plan and root hygiene
Goal: keep only one active plan in root and make the project state legible.

- [x] Archive the old root execution log.
- [x] Archive the previous root master-plan snapshot.
- [ ] Add a small regression check that root does not accumulate extra `PROJECT_MASTER_*` files again.
- [ ] Decide whether root status/progress logs are permanently forbidden or allowed only inside `docs/plan_archive/`.

Acceptance:
- root contains only `PROJECT_MASTER_PLAN.md` as the active planning file.
- archive rules are documented and followed.

### W1. Post-restore technical closure
Goal: close the real technical branch that is still red after the workspace recovery.

Current facts:
- `paper/ieee_2026/data/step28/20260308_postrestore_ai/VERIFY_SUBMISSION_CANDIDATE.json` is red.
- `outputs/step27_baseline_20260308_postrestore_s3_fixao2/acceptance_envelopes/acceptance_envelope_summary.json` is red.
- Low-compute update (`2026-03-16`):
  - AIR56 best no-training candidate under perturbation stayed below acceptance because `eta` remained slightly negative:
    - `outputs/tune_air56_20260316_p02_g12/air56_tuning_summary.json`
    - `outputs/tune_air56_20260316_refine1/air56_tuning_summary.json`
  - AL31 best no-training candidate stayed above the `err_failures<=2` target:
    - `outputs/tune_al31_20260316_refine1/al31_tuning_summary.json`
  - AO2 current baseline remained the best config-only candidate, but still failed on tracking/errors:
    - `outputs/tune_ao2_20260316_refine1/ao2_tuning_summary.json`
  - AO2 short warm-start from the existing checkpoint did not close W1:
    - same-reward pilot improved score but kept `err_failures=2.7`:
      - `outputs/tune_ao2_20260316_warmstart_eval2/ao2_tuning_summary.json`
    - tracking-biased pilot reduced errors only to `2.3` but pushed power/eta below zero:
      - `outputs/tune_ao2_20260316_tracking_eval/ao2_tuning_summary.json`
  - Conclusion: further W1 progress now requires checkpoint-level finetuning/retraining, not more supervisor-only sweeps.

Work:
- [ ] Rebuild the missing passport block for the post-restore candidate so the package no longer skips passport checks.
- [x] Re-run low-compute AL31/AO2/AIR56 tuning under the current envelope constraints instead of relying only on the old recovered checkpoint set.
- [x] Add targeted-candidate support to the tuning tool so local refinement can be evaluated without random sweeps.
- [x] Run a short AO2 warm-start pilot from the recovered checkpoint to test whether cheap checkpoint adaptation is sufficient.
- [ ] Run explicit checkpoint-level finetuning/retraining for the failing post-restore motors (AO2 first, then AL31/AIR56 if still needed).
- [ ] Re-run baseline Step27 for the selected post-restore checkpoints.
- [ ] Re-run acceptance envelopes and identify scenario-level failures by motor:
  - AIR56: `load_step`, `speed_step`
  - AL31: `load_step`, `speed_step`, `start_stop`
  - AO2: `load_step`, `ramp`, `speed_step`, `start_stop`
- [ ] Rebuild Step28 from the corrected Step27 run.
- [ ] Re-run verify/freeze/promote for a new post-restore frozen tag.

Acceptance:
- `all_rows_pass=true`
- post-restore `acceptance_pass=true` for all three motors
- `ready_for_submission=true`
- `verification_ok=true`

Artifacts expected:
- `outputs/step27_baseline_<new_tag>/`
- `outputs/step27_baseline_<new_tag>/acceptance_envelopes/`
- `paper/ieee_2026/data/step28/<new_tag>/`
- `paper/ieee_2026/data/release/<new_tag>/`

### W2. Universal any-motor algorithm closure
Goal: move from "pipeline exists" to "algorithm is convincingly reusable for a new motor".

Current facts:
- correctness gate on the demo flow is green only when the energy threshold is disabled
- energy threshold fails on AO2 in the current demo run
- the flow has not been closed on real identification data

Work:
- [ ] Separate and document two official gates:
  - correctness gate
  - energy gate
- [ ] Define the canonical verification set for the any-motor flow:
  - synthetic/demo passport
  - benchmark trio (`air56`, `al31`, `ao2`)
  - at least one identification-first run
- [ ] Close the energy gate on the verification set without disabling it.
- [ ] Prove the revalidation path without retraining:
  - `--skip-training`
  - `--init-checkpoint`
  - benchmark search over `id_ref_alpha` and `delta_id_max`
- [ ] Run the identification-first scenario and save the resulting report as a stable reference artifact.
- [ ] Decide and document whether energy acceptance is mandatory for every benchmark motor or only for the designated verification subset.

Acceptance:
- one documented flow passes correctness gate
- one documented flow passes energy gate
- one documented flow uses identification data
- the final report clearly distinguishes "control correctness" vs "energy improvement"

Artifacts expected:
- `outputs/train_any_motor_pipeline/<tag>/any_motor_onboarding_report.json`
- `outputs/train_any_motor_pipeline/<tag>/benchmark_search_summary.json`
- `outputs/train_any_motor_pipeline/<tag>/training_attempts.json`

### W3. Cross-motor generalization and non-retraining proof
Goal: prove the algorithm is not "just trained for these three motors".

Work:
- [ ] Define the held-out evaluation protocol for source vs target motors.
- [ ] Run `tools/eval_cross_motor_generalization.py` on at least one held-out split.
- [ ] Compare:
  - native checkpoint on target motor
  - transferred checkpoint from source motors
  - onboarding flow with only passport / passport+identification
- [ ] Document the gap between native and transferred/onboarded control.

Acceptance:
- one reproducible held-out report exists
- the report is referenced from the root plan and the onboarding docs
- the conclusion is explicit: where transfer is sufficient and where retraining is still required

Artifacts expected:
- `outputs/cross_motor_generalization_<tag>/cross_motor_generalization_summary.csv`
- `outputs/cross_motor_generalization_<tag>/cross_motor_generalization_report.md`

### W4. Refactor of active orchestration code
Goal: stop carrying key workflows inside giant all-in-one scripts.

Current hotspots:
- `tools/step27_pipeline.py` -> `1587` lines
- `tools/train_any_motor_pipeline.py` -> `1126` lines
- `tools/build_publication_from_markdown.py` -> `694` lines
- `tools/multi_motor_study_report.py` -> `669` lines
- `tools/train_3motors_pipeline.py` -> `640` lines
- `tools/robust_motor_hardening.py` -> `536` lines

Work:
- [ ] Extract shared benchmark evaluation and acceptance logic out of `tools/step27_pipeline.py`.
- [ ] Extract onboarding submodules out of `tools/train_any_motor_pipeline.py`:
  - passport normalization
  - optional identification loading
  - benchmark search
  - acceptance evaluation
  - report rendering
- [ ] Unify CSV/JSON/report writing helpers that are still duplicated across active pipelines.
- [ ] Keep CLI compatibility and artifact names stable while moving internals.
- [ ] Add module-level tests for extracted pure functions instead of testing only end-to-end CLIs.

Acceptance:
- active orchestration scripts become thinner wrappers around extracted modules
- outputs stay backward-compatible
- no behavior drift in smoke/regression tests

### W5. Test coverage expansion
Goal: cover the newest critical paths, not just the oldest release chain.

Work:
- [ ] Add smoke test for `tools/train_any_motor_pipeline.py` with:
  - `--skip-training`
  - external checkpoint
  - benchmark search
- [ ] Add regression test for onboarding acceptance evaluation:
  - green correctness gate
  - red energy gate
- [ ] Add regression test for benchmark search ranking selection.
- [ ] Add a small repo-hygiene test:
  - only one active root master plan
  - no stray root execution logs
- [ ] Add a documentation encoding check for `README.md` or explicitly normalize it and pin the expectation.

Acceptance:
- new onboarding behavior is protected by tests
- repo hygiene regressions fail fast
- documentation encoding regressions fail fast

### W6. Documentation and "make it beautiful" pass
Goal: make the repo readable and intentional, not just functional.

Work:
- [ ] Normalize `README.md` to UTF-8 and rewrite the entry section so it reflects the current project state.
- [ ] Update the root README quick-start to point to current supported flows:
  - frozen release reproduction
  - post-restore recovery track
  - any-motor onboarding
- [ ] Refresh `docs/runbook_3motors_ops.md` so it matches the real current critical path.
- [ ] Refresh `docs/project_structure.md` after the refactor.
- [ ] Make the status of each major track explicit:
  - stable frozen release
  - post-restore active research track
  - universal onboarding track

Acceptance:
- new contributor can understand the real state of the project from root docs
- no visibly broken encoding in the main docs
- runbooks and active CLI entrypoints match reality

## Execution order
This is the strict order for finishing the project without thrashing.

1. W0 root hygiene and archive discipline.
2. W1 post-restore red branch to green Step27/envelope/Step28.
3. W2 close the any-motor energy gate and identification-first flow.
4. W3 prove held-out generalization and non-retraining claims.
5. W4 refactor the large orchestration scripts while preserving artifact contracts.
6. W5 add tests for the newly extracted logic and new onboarding modes.
7. W6 perform the final documentation and presentation cleanup.
8. Re-run `pytest -q`.
9. Update this file with final statuses.

## Immediate next actions
- [ ] Start with post-restore envelope closure because it is the largest remaining red branch.
- [ ] In parallel, prepare the onboarding identification-first reference run.
- [ ] After both are green, refactor `tools/train_any_motor_pipeline.py` and `tools/step27_pipeline.py`.

## Update rule
- Only this file may serve as the active root master plan.
- Every update must record:
  - what was closed
  - what remains blocked
  - which artifact proves the claim
- If a track is intentionally retired instead of finished, document that decision explicitly here.
