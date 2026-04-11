# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-04-12`
Repository: `C:\mic_theory`
Last pushed commit: `34106a9`

## Scope decision on `2026-04-12`
- The active release scope is officially reduced from `3` motors to `2` motors:
  - `AIR56`
  - `AL31`
- `AO2` is preserved as a research backlog and must remain in the repository together with its configs, outputs, and publication trail.
- From this point on:
  - release closure
  - submission packaging
  - strict verify
  are judged on `AIR56 + AL31`
- `AO2` is no longer a blocker for the active release, but it must stay documented so the team can return to it later.

## Canonical source
- This file is the only active master plan allowed in the repository root.
- Historical root plans and execution logs must stay only under `docs/plan_archive/`.
- `C:\mic_theory` is the only canonical working repository.
- `C:\mt` and `C:\mic_theory_repo_restored` are not canonical roots.

## Current factual snapshot
- Git state is currently dirty because the active root plan, README, Step28 report-builder logic, and fresh `2026-04-12` 2-motor release artifacts are not pushed yet.
- Latest confirmed smoke in the current cycle is green:
  - `pytest -q tests/test_root_hygiene_smoke.py tests/test_report_plan_completion_smoke.py`
  - `3 passed`
- Latest confirmed full repository regression after the `2026-04-12` 2-motor release closure is green:
  - `pytest -q`
  - `193 passed`
- Latest confirmed focused trainer regression after the `AIR56` scenario-path fix is also green:
  - `pytest -q tests/test_train_ai_id_ref_external_step27.py`
  - `25 passed`
- Latest confirmed PPO hardening regression for reserve control-family probes is green:
  - `pytest -q tests/test_ppo_voltage_anchor.py`
  - `2 passed`
- Frozen release is already green and must not be reopened unless a real bug is found:
  - `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/VERIFY_SUBMISSION_CANDIDATE.json`
  - `verification_ok=true`
- Universal onboarding is implemented and correctness-gated, but not energy-closed:
  - this statement is no longer current for the active `2-motor` scope
  - canonical passport-only green proof:
    - `outputs/train_any_motor_pipeline/eval_2motor_rawskip_al31_20260412/any_motor_onboarding_report.json`
    - `all_ok=true`
  - canonical passport + identification green proof:
    - `outputs/train_any_motor_pipeline/eval_2motor_identskip_al31_20260412/any_motor_onboarding_report.json`
    - `all_ok=true`
  - both runs use the active benchmark scope:
    - `air56,al31`
  - `AO2` remains optional backlog validation only
- Active 2-motor release closure is now green end-to-end:
  - package:
    - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release`
  - checklist:
    - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release/FINAL_CHECKLIST_AUTO.md`
    - `ready_for_submission=true`
  - submission candidate:
    - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release/SUBMISSION_CANDIDATE.json`
    - `ready_for_submission=true`
  - strict verify:
    - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release/VERIFY_SUBMISSION_CANDIDATE.json`
    - `verification_ok=true`
- The `AL31` blocker was resolved by fixing a release-summary spec mismatch:
  - `tools/build_motor_tuning_reports_from_step28.py` no longer injects an undocumented aggregate `eta >= 0` guard for generic motors
  - scenario-level `eta/current/start_stop` constraints remain enforced by canonical Step27 acceptance envelopes
  - cross-motor checklist guardrails remain power-threshold based, aligned with:
    - `paper/ieee_2026/guardrails_policy.json`
- Historical green candidate `20260304_al31_robust_rand009_nodrift_v3` is not direct W1 proof:
  - it was built with `seed_perturbation=false`
  - it is useful only as provenance/recovery context, not as current strict `p0.2` closure evidence
- `AO2` is no longer a release blocker for the active scope; it remains a preserved research backlog.
- Current codebase now includes new reward-alignment and training-basin controls:
  - `mic_ai/ai/ai_env.py`
  - added soft energy reward gate for `ai_id_ref` / `ai_current`
  - added running-eta penalty / episode-energy knobs
  - this is code-complete and tested
  - runtime validation is now partially complete: reward-only micro-runs were validated and found insufficient
  - `mic_ai/ai/train_ai_id_ref.py`
  - added energy curriculum, CLI reward overrides, hidden-size override, hidden-size inference from warm-start checkpoints, mixed-width checkpoint adaptation, and external Step27 resume plumbing
  - now also supports per-scenario training-time overrides for:
    - reward weights
    - `ai_id_speed_tol` / `ai_id_speed_tol_rel`
    - `id_ref_gate_*`
  - scenario override JSON loading is now BOM-safe, so PowerShell-created UTF-8 files are accepted without manual cleanup
  - now also exposes `foc_assist` / `ai_speed` as trainable control modes instead of hard-limiting the CLI to `ai_id_ref` / `ai_current`
  - now also propagates `foc_assist` reward knobs from env configs and allows external Step27 scans for non-`ai_id_ref` modes without `candidate_json`
  - now also remaps `1-action ai_id_ref` warm-start checkpoints into the `id` slot when crossing into `2-action` modes (`ai_current` / `ai_speed` / `foc_assist`), instead of silently placing that policy into the `iq` slot
  - `mic_ai/ai/agents/ppo_voltage.py`
  - added optional actor-anchor penalty against the warm-start policy
  - this is code-complete and covered by a dedicated regression
  - numeric reserve-mode runtime hardening is now extended beyond core-loss:
    - safe mechanical power in `mic_ai/analysis/metrics.py`
    - safe float32 observation casting in `mic_ai/ai/ai_env.py`
    - safe current RMS path in `mic_ai/ai/ai_env.py`

## What is still not finished to 100%
- Nothing remains blocking `100%` for the active `2-motor` project scope.
- Remaining work is backlog only:
  1. optional onboarding expansion back to `AO2`
  2. optional orchestration refactor for script modularity
  3. optional future `AO2` runtime/runtime-dispatch research

## Definition of 100% done
The project is finished only when all items below are true.

- A new post-restore candidate exists with:
  - `ready_for_submission=true`
  - `checklist_ready_for_submission=true`
  - `verification_ok=true`
  - canonical tag:
    - `20260412_postrestore_ai_2motors_release`
- W1 strict closure is green for the active release motors:
  - `AIR56` green on mean and worst-case criteria
  - `AL31` green on mean and worst-case criteria
- `AO2` remains archived as a documented research backlog with preserved artifacts and an explicit return path.
- Universal onboarding has:
  - benchmark correctness green
  - energy gate green
  - documented passport-only flow
  - documented passport + identification flow
  - reproducible artifacts for both
- Documentation is clean and current:
  - `README.md` readable in UTF-8
  - `README.md` explicitly states that the release scope is `AIR56 + AL31`
  - `README.md` explicitly states that `AO2` is paused research, not deleted
  - runbooks match actual entrypoints and gates
  - only one active root plan remains
- `pytest -q` is green after all cleanup.
  - current green reference:
    - `193 passed`

Status on `2026-04-12`:
- all mandatory conditions above are satisfied for the active `2-motor` scope
- the project is considered complete for:
  - `AIR56`
  - `AL31`
- anything involving `AO2` from this point is backlog research, not an active completion blocker

## Do not do
- Do not weaken acceptance thresholds to force a green result.
- Do not reopen `20260303_ai_config_locked_nodrift` for cosmetic changes.
- Do not create new dated root plans.
- Do not spend compute on repeated candidate-only sweeps already proven exhausted.
- Do not run full Step28 packaging again until W1 strict closure is actually green.

## Compute policy
Because training is slow and the machine is weak, research must proceed in this order:
1. cheapest experiment that can change the strict blocker
2. only then medium warm-start
3. only then full scratch / rebuild
4. full Step28 reproduce only after motor-level strict closure is green

## Active workstreams

### W0. Root hygiene
Goal: keep the project state legible.

Status:
- root already contains a single active plan
- historical plans are archived
- root hygiene regression is now added:
  - `tests/test_root_hygiene_smoke.py`
  - paired smoke remains green together with the plan-completion smoke:
    - `pytest -q tests/test_root_hygiene_smoke.py tests/test_report_plan_completion_smoke.py`
    - `3 passed`

Remaining:
- document that progress logs live only in `docs/plan_archive/`

Acceptance:
- root contains only `PROJECT_MASTER_PLAN.md` as active planning file
- hygiene rule is protected by test or smoke check

### W1. Post-restore technical closure
Goal: close the real red branch that still blocks submission.

Status on `2026-04-12`:
- done for the active `2-motor` release scope
- canonical green package:
  - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release`
- canonical green verify:
  - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release/VERIFY_SUBMISSION_CANDIDATE.json`
  - `verification_ok=true`
- `AO2` stays below as backlog/provenance only and must not be deleted

### W2. Universal onboarding
Goal: prove onboarding for the active `2-motor` scope.

Status on `2026-04-12`:
- done for the active `2-motor` scope
- passport-only green artifact:
  - `outputs/train_any_motor_pipeline/eval_2motor_rawskip_al31_20260412/any_motor_onboarding_report.json`
- passport + identification green artifact:
  - `outputs/train_any_motor_pipeline/eval_2motor_identskip_al31_20260412/any_motor_onboarding_report.json`
- active benchmark scope is now aligned with the release scope:
  - `air56,al31`
- `AO2` can still be added explicitly via:
  - `--benchmark-motors air56,al31,ao2`
  but it is backlog-only and not part of current completion.

### W3. Backlog after completion
These items are preserved, but they are not blockers for `100%` completion of the active `2-motor` project.

1. Refactor monolithic orchestration scripts into smaller helpers.
2. Re-open onboarding benchmark scope to include `AO2`.
3. Continue `AO2` runtime/hybrid research from the preserved backlog artifacts.

#### W1.0 Runtime correction on `2026-04-11`

- This historical narrative remains archived here only as provenance for the removed `3-motor` scope.
- It is no longer the active release blocker path after the `2026-04-12` scope reduction to `AIR56 + AL31`.
- Current runtime facts:
  - canonical no-perturb live `Step27` for all 3 motors is green:
    - `outputs/step27_3motors_hybrid_verify_fix2_20260411ar/step27_report.md`
  - latest full strict reproduce candidate is still red only because of `AO2`:
    - `paper/ieee_2026/data/step28/20260411_postrestore_ai_hybrid_final/FINAL_CHECKLIST_AUTO.md`
  - `AIR56` strict `p0.2`: green in the same current runtime
  - `AL31` strict `p0.2`: green in the same current runtime
  - `AO2` no-perturb live runtime is green with the refreshed deploy pair in:
    - `config/env_research_ao2_32_4_3kw.py`
- Current best strict `AO2 p0.2` canonical envelope frontier is no longer the old `20260411az` lineage.
- The current strict AO2 state after the full `2026-04-11` evening replay sequence is:
  - best canonical-selector incumbent still has `2` failing rows concentrated in `start_stop`:
    - checkpoint/candidate:
      - `actor_ep002 + bridge_mid_03`
    - artifact:
      - `outputs/ao2_p02_medium_rebuild_train_20260411bk/20260411_155245_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan_shortlist/ao2_checkpoint_scan_summary.json`
    - exact metrics:
      - `avg_power_saving_pct = 0.9235132823428183`
      - `avg_eta_gain_pct = 0.4572220931523557`
      - `err_failures = 0.2`
      - `start_stop_power_saving_pct = 3.394883880813071`
      - `worst_current_peak_ratio = 1.2034716634303517`
      - `avg_power_saving_pct_min_seed = 0.3374281115906236`
      - `avg_eta_gain_pct_min_seed = -2.9561571988572375`
      - `start_stop_power_saving_pct_min_seed = 1.087047629079596`
      - `envelope_fail_count = 2`
      - `envelope_scenario_fail_count = 1`
      - `envelope_gap_total = 7.336741262359363`
      - `envelope_eta_gap = 6.336741262359363`
      - `envelope_err_fail_count = 1`
  - new lower-gap AO2 secondary frontier exists, but it adds one extra `speed_step` row fail:
    - checkpoint/candidate:
      - `actor_ep003 + bridge_mid_03`
    - artifact:
      - `outputs/ao2_p02_medium650_softgate_rebuild_20260411bn/20260411_165450_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - exact metrics:
      - `avg_power_saving_pct = 0.9126615068913875`
      - `avg_eta_gain_pct = 0.7660685588469006`
      - `err_failures = 0.2`
      - `start_stop_power_saving_pct = 3.6167329868708498`
      - `avg_power_saving_pct_min_seed = 0.3435855743208538`
      - `avg_eta_gain_pct_min_seed = -2.475759296629143`
      - `envelope_fail_count = 3`
      - `envelope_scenario_fail_count = 2`
      - `envelope_gap_total = 5.9100665284304394`
      - `envelope_power_gap = 0.07025220837094714`
      - `envelope_eta_gap = 4.839814320059492`
      - `envelope_err_fail_count = 1`
  - a new wider-basin `96x96` continuation also produced a viable but not better strict frontier:
    - checkpoint/candidate:
      - `actor_ep000 + exit_975`
    - artifact:
      - `outputs/ao2_ep003_speedfix_capacity96_20260411bq/20260411_175040_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - exact metrics:
      - `avg_power_saving_pct = 1.1840396376131974`
      - `avg_eta_gain_pct = 0.7516484564727538`
      - `err_failures = 0.2`
      - `start_stop_power_saving_pct = 4.003258461661492`
      - `avg_power_saving_pct_min_seed = 0.496536309696205`
      - `avg_eta_gain_pct_min_seed = -2.9260353005523903`
      - `envelope_fail_count = 2`
      - `envelope_scenario_fail_count = 1`
      - `envelope_gap_total = 7.333020516135628`
      - `envelope_eta_gap = 6.333020516135628`
      - `envelope_err_fail_count = 1`
- What the latest AO2 sequence proved:
  - the interrupted medium rebuild scan (`bk`) was worth finishing and became the best strict canonical incumbent
  - local candidate search around `actor_ep002 + bridge_mid_03` is exhausted:
    - `outputs/ao2_ep002_bridge_local_20260411bl/ao2_tuning_summary.json`
  - corrected full-horizon continuation with soft energy gate did not beat the incumbent but exposed `actor_ep003` as the best lower-gap secondary frontier:
    - `outputs/ao2_p02_medium650_softgate_rebuild_20260411bn/.../ao2_checkpoint_scan_summary.json`
  - local candidate search around `actor_ep003 + bridge_mid_03` is also exhausted:
    - `outputs/ao2_ep003_local_20260411bo/ao2_tuning_summary.json`
  - low-lr speed-fix continuation from `actor_ep003` did not close strict replay:
    - `outputs/ao2_ep003_speedfix_train_20260411bp/.../ao2_checkpoint_scan_summary.json`
  - wider-basin `96x96` continuation from `actor_ep003` did not close strict replay either:
    - `outputs/ao2_ep003_speedfix_capacity96_20260411bq/.../ao2_checkpoint_scan_summary.json`
  - local candidate search around the `96x96 actor_ep000 + exit_975` frontier is exhausted:
    - `outputs/ao2_96x96_ep000_exit975_local_20260411br/ao2_tuning_summary.json`
  - true `96x96` scratch / rebuild is also now explicitly red:
    - `outputs/ao2_scratch96_strict_20260411bs/20260411_182011_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - observed result:
      - all evaluated checkpoints stayed far below strict closure
      - power/start_stop collapsed negative
      - scratch `96x96` is not a justified repeat branch on weak hardware
  - generic `foc_assist` probe no longer crashes after PPO `NaN/Inf` hardening, but the actual AO2 control family result is red enough to reject as a near-pass branch:
    - runtime bugfix now landed in:
      - `mic_ai/ai/agents/ppo_voltage.py`
      - `tests/test_ppo_voltage_anchor.py`
    - probe artifact:
      - `outputs/ao2_focassist_smoke_20260411bt2/20260411_184706_env_research_ao2_32_4_3kw_foc_assist/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - observed result:
      - `avg_power_saving_pct = 100.0`
      - `avg_eta_gain_pct = -100.0`
      - `err_failures = 4.0`
      - branch is numerically alive but physically invalid for AO2 strict closure
  - deterministic failing-seed replay is now implemented in the trainer and already proved useful for AO2 strict closure:
    - trainer capability now exists in:
      - `mic_ai/ai/train_ai_id_ref.py`
      - `tests/test_train_ai_id_ref_external_step27.py`
    - first validated replay branch:
      - `outputs/ao2_seedreplay_202_505_20260411bu/20260411_193243_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - best replay frontier:
      - `actor_ep003 + bridge_mid_03`
    - exact metrics:
      - `avg_power_saving_pct = 1.003175908685265`
      - `avg_eta_gain_pct = 0.6687548796177822`
      - `err_failures = 0.2`
      - `start_stop_power_saving_pct = 3.603707331359078`
      - `worst_current_peak_ratio = 1.2100318664241658`
      - `avg_power_saving_pct_min_seed = 0.4551386548973607`
      - `avg_eta_gain_pct_min_seed = -2.8122709901745506`
      - `envelope_fail_count = 2`
      - `envelope_gap_total = 6.732672625849837`
      - `acceptance_pass_aggregate = true`
      - `acceptance_pass = false`
    - meaning:
      - this beats the old canonical incumbent from `20260411bk` on strict envelope gap without adding a new scenario fail
      - the remaining blocker is still concentrated in `start_stop`
      - the AO2 task was previously under-specified because mean reward optimization was not directly replaying the failing strict seeds
  - the next weighted replay continuation was also checked and did not improve the replay frontier:
    - branch:
      - `outputs/ao2_seedreplay_505heavy_20260411bv/20260411_195139_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan_detached/ao2_checkpoint_scan_summary.json`
    - training recipe:
      - warm-start from `outputs/ao2_seedreplay_202_505_20260411bu/.../eval/actor_ep003.pth`
      - short `505,505,505,202,202` episode-seed cycle
      - same strict shortlist:
        - `exit_975`
        - `p02fix09_base`
        - `bridge_mid_03`
        - `local_idle_lessneg`
    - result:
      - selector best = `actor_ep_init + bridge_mid_03`
      - metrics are identical to the original `20260411bu` replay frontier
      - no new checkpoint beat the replay incumbent
    - meaning:
      - replaying failing seeds is directionally correct
      - but the current `ai_id_ref 64x64` basin is now exhausted even under weighted replay
- the remaining blocker is now tightly defined:
  - one `start_stop` eta worst-seed tail
  - one `start_stop` error/tracking fail
  - and, on the lower-gap `actor_ep003` branch, one extra tiny `speed_step` power fail
- Active execution order from this point:
  1. Do not reopen `AIR56` or `AL31` unless the next final full run regresses them.
  2. Spend the next compute only on `AO2`.
  3. Do not spend more compute on local candidate grids around:
     - `actor_ep002 + bridge_mid_03`
     - `actor_ep003 + bridge_mid_03`
     - `96x96 actor_ep000 + exit_975`
  4. The next justified AO2 step is no longer another `ai_id_ref` weighted replay micro-run.
     - weighted replay already stalled at `20260411bv`
     - do not repeat more `64x64 ai_id_ref` micro-continuations around the same replay frontier
     - do not repeat `96x96` scratch
     - do not spend more compute on `AO2 foc_assist` until its reward / physics path is separately rehabilitated
     - next cheapest materially different path is:
       - `AO2 ai_speed` reserve probe with the same strict `p0.2` evaluation discipline
     - if `ai_speed` is also red, only then move to:
       - a larger-basin full-horizon rebuild (`128x128`) if extra compute is acceptable
     - only after that, at most one targeted runtime retune on the winning checkpoint
  5. `AO2 ai_speed` reserve probe has now also been checked and is red:
     - train artifact:
       - `outputs/ao2_aispeed_seedreplay_20260411bw/20260411_202821_env_research_ao2_32_4_3kw_ai_speed/training_metrics.json`
     - strict scan artifact:
       - `outputs/ao2_aispeed_seedreplay_20260411bw/20260411_202821_env_research_ao2_32_4_3kw_ai_speed/external_step27_scan/ao2_checkpoint_scan_summary.json`
     - observed result:
       - every evaluated checkpoint stayed at `power=100`, `eta=-100`, `err_failures=4.0`
       - reserve family is numerically alive but physically invalid for AO2 strict closure
  6. Therefore the next justified AO2 path is now uniquely defined:
     - `ai_id_ref` larger-basin full-horizon rebuild (`128x128`)
     - keep the failing-seed replay formulation because it was the only thing that actually reduced canonical strict gap
     - return to balanced replay of `202` and `505` instead of the already-stalled `505`-heavy micro-run
  7. `2026-04-11` late-evening AO2 correction:
     - the real canonical incumbent is now the `128x128` replay rebuild, not the older `64x64` replay frontier:
       - artifact:
         - `outputs/ao2_seedreplay_capacity128_20260411bx/20260411_203804_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
       - best pair:
         - `actor_ep001 + bridge_mid_03`
       - exact metrics:
         - `avg_power_saving_pct = 1.0474133184923837`
         - `avg_eta_gain_pct = 0.6641948627646582`
         - `err_failures = 0.2`
         - `start_stop_power_saving_pct = 3.577738330411657`
         - `worst_current_peak_ratio = 1.1826221055084283`
         - `avg_power_saving_pct_min_seed = 0.3448140237010511`
         - `avg_eta_gain_pct_min_seed = -2.4484040804927054`
         - `envelope_fail_count = 2`
         - `envelope_gap_total = 5.4913308513227435`
         - `envelope_eta_gap = 4.4913308513227435`
         - `envelope_err_fail_count = 1`
     - targeted runtime retune around this incumbent is exhausted:
       - `outputs/ao2_capacity128_ep001_localsearch_20260411by/ao2_tuning_summary.json`
       - best remained `bridge_mid_03`
     - low-lr continuation from the new incumbent is also a dead end:
       - `outputs/ao2_capacity128_ep001_continue_20260411bz/20260411_213613_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
       - strict gap regressed versus `bx`
     - first mixed seed-aware continuation is also a dead end:
       - `outputs/ao2_capacity128_seedaware_20260411cb/20260411_224529_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
       - selector best fell back to `actor_ep000 + bridge_mid_03`
       - strict gap regressed to `7.817854367269087`
     - second cheap mixed start-stop continuation is also a dead end:
       - `outputs/ao2_capacity128_seedfix2_20260411cd/20260411_232730_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
       - best new checkpoint `actor_ep003 + bridge_mid_03`
       - strict gap still regressed to `6.612008644217941`
     - focused seed-pair supervisor probe proved the current blocker is not solvable by a static deploy candidate alone:
       - `outputs/ao2_bridge_seedpair_probe_20260411cc/ao2_tuning_summary.json`
       - the only candidate that removed the `505` tracking fail on the `{202,505}` probe did so by collapsing power/eta and violating strict current/efficiency margins
       - the full strict incumbent remained `bridge_mid_03`
     - runtime `ai_id_ref_hybrid` probe with state-dependent primary/secondary switching also had no effect on the failing pair:
       - aggressive secondary supervisor did not change the `202` / `505` raw `start_stop` outcomes relative to `bx`
       - meaning:
         - the current load-delta hybrid trigger is not the missing piece for AO2 strict closure
     - raw diagnosis of the current incumbent is now sharp:
       - `seed 202` is an eta-only failure on `start_stop`
       - `seed 505` is a tracking/error-only failure on `start_stop`
       - no checked existing checkpoint fixed `505` under `bridge_mid_03`
       - therefore the mixed replay formulation is still under-constrained even with seed-specific overrides
  8. The next justified AO2 path is now updated again:
     - do not spend more compute on mixed `202+505` micro-runs around the same `bx` basin
     - do not spend more compute on candidate-only sweeps around `bridge_mid_03`
     - do not spend more compute on `ai_id_ref_hybrid` unless its trigger logic is redesigned
     - next step must be sequential, not mixed:
       - Phase A: close `seed 505 / start_stop / err_ok` with a short tracking-first continuation from `bx actor_ep001`
       - Phase B: only after `505` is fixed, run a separate eta-recovery continuation for `seed 202 / start_stop`
     - only after a Phase A / Phase B pair produces a new strict winner should full `Step27 -> Step28` be rerun
  9. `2026-04-12` overnight AO2 Phase-A results:
     - short anchored `505-only` continuation did not fix `505`:
       - `outputs/ao2_capacity128_phaseA_505track_20260411ce/20260411_235656_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
       - `505 err` remained red and the global strict gap regressed versus `bx`
     - stronger medium `505-only` continuation also failed the cheap gate:
       - gate artifact:
         - `outputs/ao2_capacity128_phaseA_505track_medium_20260411cf/20260412_001731_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan_seed505_gate/ao2_checkpoint_scan_summary.json`
       - every evaluated checkpoint still had `err_failures = 1.0` on `seed=505, scenario=start_stop`
       - meaning:
         - the incumbent warm-start basin cannot repair `505` even when isolated
     - short `505-only` scratch did produce a true `505`-fix policy:
       - gate artifact:
         - `outputs/ao2_phaseA_505track_scratch_20260412a/20260412_002036_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan_seed505_gate/ao2_checkpoint_scan_summary.json`
       - best gate checkpoint:
         - `actor_ep002 + bridge_mid_03`
       - gate metrics on `seed=505,start_stop`:
         - `avg_power_saving_pct = 13.24249731719236`
         - `avg_eta_gain_pct = 12.767558727107332`
         - `err_failures = 0.0`
         - `acceptance_pass = true`
     - but that same `505`-fix policy is not globally deployable as a monolithic controller:
       - full strict recheck:
         - `outputs/ao2_phaseA_505track_scratch_20260412a/20260412_002036_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_scan_ep002_fullstrict/ao2_checkpoint_scan_summary.json`
       - result:
         - `avg_power_saving_pct = 1.987%`
         - `avg_eta_gain_pct = -1.577%`
         - `err_failures = 1.6`
         - `start_stop_power_saving_pct = 0.372%`
         - meaning:
           - the project now has direct evidence that `505` can be solved by policy capacity / basin shift
           - but the resulting policy is too specialized to replace the incumbent globally
  10. `2026-04-12` runtime dispatch conclusion:
      - a new hybrid trigger helper now exists in:
        - `tools/step27_pipeline.py`
        - `tests/test_step27_hybrid_trigger.py`
      - hybrid switching by raw speed-error threshold was probed with:
        - primary = `bx actor_ep001`
        - secondary = `20260412a actor_ep002`
      - result:
        - current hybrid dispatch remains unusable
        - probes either stayed red on tracking or became numerically/physically unstable
      - meaning:
        - the new evidence does not support more compute on the current simple hybrid trigger
        - the next justified path is now narrower:
          - either redesign hybrid dispatch / observability beyond `load_delta` and simple `speed_err` thresholding
          - or move to a richer monolithic policy family that can internalize both regimes without runtime switching

#### W1.1 Current motor status

`AO2`
- status: green only in no-perturb live runtime; red in strict `p0.2`
- blocker status: open
- active blocker: `start_stop` worst-seed eta/error tail under strict `p0.2`

`AL31`
- current best deploy pair:
  - checkpoint: `outputs/al31_anchor_ep008_medium4_20260328a/results_run/20260328_110425_tmp_al31_mid04_train_20260322_ai_id_ref/eval/actor_ep_init.pth`
  - candidate: `mid04_speed_dn_04`
  - artifact: `outputs/al31_mid04_ultrafine2_20260328l/al31_tuning_summary.json`
- exact current metrics:
  - `avg_power_saving_pct = 1.0482255792898605`
  - `avg_eta_gain_pct = 0.006584767081225795`
  - `avg_power_saving_pct_min_seed = 0.3880779822114183`
  - `avg_eta_gain_pct_min_seed = -0.00020601519861440654`
  - `envelope_all_rows_pass = true`
- meaning:
  - mean gate is already green
  - row-level envelope is green
  - only remaining blocker is a tiny worst-case eta tail
- already proven dead ends:
  - wider bridge-search around `actor_ep005`
  - micro-finetune with old hard reward gate
  - high-speed eta-biased micro-finetune with old hard reward gate
  - true soft-gate rerun after config-propagation fix
  - running-eta reward micro-run
  - terminal-energy-bonus micro-run
  - valid two-stage `64x64` base-config run with strict external Step27 selector
  - targeted retune around the best valid `64x64` two-stage checkpoint
  - targeted local-safe search around the best `96x96` checkpoint `actor_ep026`
  - valid actor-anchor micro-run with low sigma and strict resume-completed external scan:
    - `outputs/al31_anchor_eta_micro2_lowsigma_20260323a/results_run/20260323_131259_env_research_al31_4_06kw_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
  - local retune around the new best checkpoint `actor_ep008`:
    - `outputs/al31_ep008_localsearch_20260326a/al31_tuning_summary.json`
  - local retune around nearby checkpoint `actor_ep000`:
    - `outputs/al31_ep000_localsearch_20260326a/al31_tuning_summary.json`
  - aligned micro-finetune on `tmp_al31_mid04_train_20260322.py` from `actor_ep008`:
    - `outputs/al31_aligned_micro3_20260326a/results_run/20260326_100303_tmp_al31_mid04_train_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
- current conclusion:
  - `AL31` cheap closure improved again after the later ultrafine replay on top of the medium anchored incumbent:
    - best pair is now `actor_ep_init + mid04_speed_dn_04`
  - the remaining strict blocker is now extremely small and purely aggregate (`avg_eta_gain_pct_min_seed = -0.000206`)
  - further cheap local retunes and aligned micro-finetune did not close it
  - merged strict candidate replay around `actor_ep008` also did not beat `al31_mid_04`:
    - `outputs/al31_ep008_merged_strict_20260326/al31_tuning_summary.json`
  - medium-budget anchored run from `actor_ep008` also completed and re-selected the warm-start incumbent instead of a new checkpoint:
    - `outputs/al31_anchor_ep008_medium4_20260328a/results_run/20260328_110425_tmp_al31_mid04_train_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - result:
      - `avg_power_saving_pct = 0.6211151560032363`
      - `avg_eta_gain_pct = 0.006371654887438738`
      - `avg_power_saving_pct_min_seed = 0.35404002065272944`
      - `avg_eta_gain_pct_min_seed = -0.00034895129392975566`
      - `envelope_all_rows_pass = true`
      - selector best = `actor_ep_init` (same incumbent weights, no new winner)
- current conclusion after that run:
  - `AL31 ai_id_ref` medium-budget continuation is now also exhausted
  - the remaining blocker is still the same tiny worst-case eta tail
  - do not spend more compute on another near-identical `ai_id_ref` branch until `AIR56` is resolved and final full-run context is available
  - dense local-safe ultrafine replay around the incumbent also completed without lifting the tail above zero:
    - `outputs/al31_ultrafine3_dense_20260411a/al31_tuning_summary.json`
    - best remained baseline / `mid04_speed_dn_04`
    - this closes the current cheap runtime-search path for `AL31`

`AIR56`
- current strict incumbent pair:
  - checkpoint: `outputs/air56_ep003_tailfocus_micro2_20260322/results_run/20260322_131130_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep005.pth`
  - candidate: `mix04_base`
  - artifact: `outputs/air56_ep003_tailfocus_micro2_20260322/results_run/20260322_131130_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
- exact incumbent metrics:
  - `avg_power_saving_pct = 0.6147884835796003`
  - `avg_eta_gain_pct = 0.0013968558560217836`
  - `avg_power_saving_pct_min_seed = 0.5151508570179902`
  - `avg_eta_gain_pct_min_seed = -0.019478794161115198`
  - `envelope_all_rows_pass = true`
  - `err_failures = 0.0`
- current closest near-pass frontier:
  - checkpoint: `outputs/air56_loadfix_etaep_20260328o/results_run/20260328_220228_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep_init.pth`
  - candidate: `etaep_bias_dn_2`
  - artifact: `outputs/air56_etaep_shortlist_top3_20260328q/air56_checkpoint_scan_summary.json`
  - exact frontier metrics:
    - `avg_power_saving_pct = 0.5968216082041533`
    - `avg_eta_gain_pct = 0.007700149746570628`
    - `avg_power_saving_pct_min_seed = 0.46497394850734397`
    - `avg_eta_gain_pct_min_seed = -0.00990541936307654`
    - `err_failures = 0.2`
    - `envelope_all_rows_pass = false`
- meaning:
  - the incumbent strict pair is still the only fully envelope-clean `AIR56` baseline, but its eta tail is still too negative
  - the best current near-pass frontier is no longer the older reltrack `speed_step` frontier
  - the active blocker has narrowed further:
    - one canonical `load_step` error row remains red
    - the near-pass frontier still fails worst-case aggregate eta
- already proven dead ends:
  - candidate-only sweep around the best checkpoint
  - low-lr eta-edge warm-start with old hard reward gate
  - high-speed/high-load warm-start with old hard reward gate
  - longer eta-biased warm-start with old hard reward gate
  - high-speed scratch rebuild with old hard reward gate
  - valid two-stage `64x64` base-config run with strict external Step27 selector
  - relaxed `Phase A` power-guard curriculum with detached post-hoc rescan
  - running-eta reward micro-run
  - terminal-energy-bonus micro-run
  - running-eta reward with explicit `eta_episode_norm` observation
  - actor-anchor eta micro-run on the strict baseline lineage
  - reltrack-lineage checkpoint scan with `mix04_rand_015`:
    - `outputs/air56_reltrack_lineage_scan_mix04_rand015_20260326a/air56_checkpoint_scan_summary.json`
  - reltrack-lineage checkpoint scan with `mix04_rand_019`:
    - `outputs/air56_reltrack_lineage_scan_mix04_rand019_20260326a/air56_checkpoint_scan_summary.json`
  - reltrack-lineage checkpoint scan with `mix04_rand_007`:
    - `outputs/air56_reltrack_lineage_scan_mix04_rand007_20260326a/air56_checkpoint_scan_summary.json`
  - local retune around `AIR56 actor_ep004 + mix04_rand007`:
    - `outputs/air56_reltrack_ep004_localsearch_20260326b/air56_tuning_summary.json`
  - micro-finetune from `AIR56 actor_ep004` on `pin60hb` config with external `rand007_soft_track` selector:
    - `outputs/air56_reltrack_ep004_microtrain_20260326c/results_run/20260326_120123_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
  - aligned `mix04` micro-finetune from the same checkpoint with the same selector:
    - `outputs/air56_reltrack_ep004_mix04_microtrain_20260326d/results_run/20260326_121057_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
  - note on the two previous micro-finetunes above:
    - their initial negative-power external selection result was partly invalidated by a mixed-feature warm-start bug
    - the init checkpoint in those runs was being evaluated with the wrong feature-space during built-in external Step27 scan
    - fixed recheck artifact:
      - `outputs/air56_reltrack_ep004_mix04_microtrain_20260326d/external_recheck_rand007_after_fix/air56_checkpoint_scan_summary.json`
  - valid medium-budget reltrack continuation after the mixed-feature fix:
    - `outputs/air56_reltrack_ep004_mediumfix_20260326e/results_run/20260326_123606_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
  - post-hoc scan of that same valid medium-budget continuation under `mix04_rand_015`:
    - `outputs/air56_reltrack_ep004_mediumfix_20260326e/posthoc_scan_rand015/air56_checkpoint_scan_summary.json`
  - checkpoint-shortlist replay on the newer eta-focused lineage:
    - `outputs/air56_etaep_shortlist_top3_20260328q/air56_checkpoint_scan_summary.json`
    - best remained `actor_ep_init + etaep_bias_dn_2`
    - still one `load_step` row fail and strict aggregate eta red
  - local search around `actor_ep_init + etaep_bias_dn_2`:
    - `outputs/air56_etaep_init_localsearch_20260328r/air56_tuning_summary.json`
    - envelope-clean candidate `etaep_loadfix_delta_1` exists, but it falls below strict aggregate power / eta thresholds
    - best near-pass candidate remained `etaep_bias_dn_2`-family and did not close strict acceptance
  - current scientific conclusion:
  - `AIR56` is no longer a pure deploy-parameter problem, and the active frontier is now eta-focused rather than the older reltrack `speed_step` branch
  - the remaining cheap blocker is still checkpoint-level, but the candidate layer around the new eta frontier is now effectively exhausted
  - shortlist-per-checkpoint ranking proved useful and is now the preferred cheap selector:
    - it surfaced `actor_ep_init + etaep_bias_dn_2` as the strongest current near-pass pair
  - short/medium load-step-focused continuation from the eta frontier with built-in shortlist external selection is now also exhausted:
    - run root:
      - `outputs/air56_etaep_loadtrain_short_20260328s`
    - newly trained checkpoints actually evaluated:
      - `actor_ep000`
      - `actor_ep001`
      - `actor_ep002`
      - `actor_ep003`
    - best newly trained snapshot:
      - `actor_ep003 + mix04_base`
      - raw metrics from the persisted progress/state files:
        - `avg_power_saving_pct = 0.5105837976366528`
        - `avg_eta_gain_pct = 0.01064860988664873`
        - `avg_power_saving_pct_min_seed = 0.24052185845599705`
        - `avg_eta_gain_pct_min_seed = -0.01202997034184805`
        - `err_failures = 0.2`
        - `envelope_all_rows_pass = false`
    - conclusion:
      - none of the new snapshots beat the already known eta frontier `actor_ep_init + etaep_bias_dn_2`
      - the only remaining unevaluated row in the interrupted resume was `actor_ep_init`, i.e. the already known frontier checkpoint rather than a new policy
      - finishing that duplicate init replay was not worth the extra compute on the weak machine
  - merged strict candidate replay around the envelope-clean incumbent also did not beat `mix04_base`:
    - `outputs/air56_ep005_merged_strict_20260326/air56_tuning_summary.json`
  - medium-budget incumbent continuation with mild energy pressure and actor anchor also did not beat the incumbent:
    - `outputs/air56_incumbent_anchor_softgate_20260326f/results_run/20260326_141652_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - result:
      - selector re-ranked `actor_ep000` above `actor_ep_init`, but both remained effectively the same strict basin
      - best new row:
        - `avg_power_saving_pct = 0.6146122654188096`
        - `avg_eta_gain_pct = 0.0013981770872645294`
        - `avg_power_saving_pct_min_seed = 0.5149501849730925`
        - `avg_eta_gain_pct_min_seed = -0.019477355927777218`
        - `envelope_all_rows_pass = true`
      - conclusion:
        - the strict incumbent basin remains stable
        - but this continuation recipe still does not move the negative worst-case eta tail
  - later strict `128x128` incumbent-basin continuation also finished without a new winner:
    - `outputs/air56_incumbent_basin128_20260329a/results_run/20260329_011801_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - best remained `actor_ep_init + mix04_base`
    - exact result:
      - `avg_power_saving_pct = 0.6147884835796003`
      - `avg_eta_gain_pct = 0.0013968558560217836`
      - `avg_power_saving_pct_min_seed = 0.5151508570179902`
      - `avg_eta_gain_pct_min_seed = -0.019478794161115198`
      - `envelope_all_rows_pass = true`
  - explicit local-safe retune around the new basin checkpoint `actor_ep019` also failed to close strict acceptance:
    - `outputs/air56_ep019_basin128_localsearch_20260408a/air56_tuning_summary.json`
    - best remained the checkpoint baseline
  - strict power-recovery continuation from `actor_ep019` produced a new aggregate-passing frontier, then was stopped early after frontier convergence:
    - `outputs/air56_ep019_powerrecover_20260408b/results_run/20260408_153717_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_state.json`
    - evaluated checkpoints through `actor_ep010`
    - result:
      - checkpoints `actor_ep001` through `actor_ep010` already pass strict aggregate gates
      - failure mode is no longer aggregate power / eta; it is row-level `load_step` tracking (`err`)
      - strongest retune base is `actor_ep006`:
        - `avg_power_saving_pct = 0.8630392380153374`
        - `avg_eta_gain_pct = 0.10702947229523918`
        - `avg_power_saving_pct_min_seed = 0.7322733162081374`
        - `avg_eta_gain_pct_min_seed = 0.08032673473104546`
        - `load_step pass_count = 2/5`
      - later checkpoints `actor_ep007` through `actor_ep010` increased aggregate margin further, but degraded `load_step` robustness:
        - best aggregate row seen was `actor_ep010`
        - `avg_power_saving_pct = 1.1987031809245041`
        - `avg_eta_gain_pct = 0.14100228475193544`
        - `avg_power_saving_pct_min_seed = 1.069864840233517`
        - `avg_eta_gain_pct_min_seed = 0.10926801078245396`
        - `load_step pass_count = 0/5`
    - current execution rule:
    - do not start another retrain from this branch
    - run one targeted local-safe retune around the best aggregate-passing checkpoint from this branch, prioritizing:
      - `load_step pass_count`
      - then `avg_power_saving_pct_min_seed`
      - then `avg_eta_gain_pct_min_seed`
    - prepared candidate set for this step:
      - `outputs/tmp_air56_ep006_loadfix_candidates_20260408c.json`
  - cheap deploy-layer closure around the new frontier is now exhausted:
    - `outputs/air56_ep006_loadfix_retune_20260408c/air56_tuning_summary.json`
    - `outputs/air56_ep003_loadfix_retune_20260408d/air56_tuning_summary.json`
    - `outputs/air56_ep002_gatepush_retune_20260408f/air56_tuning_summary.json`
    - result:
      - none of the targeted retunes lifted `load_step` above `2/5`
      - best strict aggregate remained near:
        - `actor_ep002 + gatepush_base`
        - `avg_power_saving_pct = 0.871666248176804`
        - `avg_eta_gain_pct = 0.12165800277003369`
        - `avg_power_saving_pct_min_seed = 0.688967361816506`
        - `avg_eta_gain_pct_min_seed = 0.09873716093466156`
        - `load_step pass_count = 2/5`
      - conclusion:
        - candidate-only closure for the current AIR56 frontier is exhausted
        - the next justified step is training-level continuation, not another deploy sweep
  - first training-level load-step-heavy continuation with unchanged `w_speed=1.0` was cut early as a bad direction:
    - partial branch:
      - `outputs/air56_ep002_loadheavy_multitag_20260408g/results_run/20260408_200239_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_progress.json`
    - partial result before stop:
      - `actor_ep001 + gatepush_softbias`
      - `avg_power_saving_pct = 1.0572538354551404`
      - `avg_eta_gain_pct = 0.129798548254087`
      - `avg_power_saving_pct_min_seed = 0.9594248805978312`
      - `avg_eta_gain_pct_min_seed = 0.11383264526707848`
      - `load_step pass_count = 1/5`
    - conclusion:
      - load-heavy sampling alone was pushing aggregate up while degrading the actual blocker
  - scenario-specific reward-only continuation from the current frontier is now also closed as a dead end:
    - code path:
      - `mic_ai/ai/train_ai_id_ref.py`
      - added `--scenario-reward-overrides-json`
    - run:
      - `outputs/air56_actor_ep001_scenario_override_20260411a`
    - strict partial verdict before early stop:
      - best new row after `5/9` processed checkpoints:
        - `actor_ep000 + gatepush_base`
        - `avg_power_saving_pct = 1.0061047586070822`
        - `avg_eta_gain_pct = 0.1159674136063582`
        - `avg_power_saving_pct_min_seed = 0.9434941923518431`
        - `avg_eta_gain_pct_min_seed = 0.08997925122549155`
        - `load_step pass_count = 2/5`
      - later trained checkpoints degraded to `load_step pass_count = 1/5`
    - conclusion:
      - scenario-specific reward shifts alone did not move the row-level blocker
      - keeping the run alive past `5/9` checkpoints was not justified on the weak machine
  - scenario-specific load-step gate/tolerance continuation is the current best new signal but still not a closure:
    - code path:
      - `mic_ai/ai/train_ai_id_ref.py`
      - per-scenario overrides now cover:
        - `ai_id_speed_tol`
        - `ai_id_speed_tol_rel`
        - `id_ref_gate_speed_tol`
        - `id_ref_gate_speed_tol_rel`
        - `id_ref_gate_min_scale`
        - `id_ref_gate_exponent`
    - run:
      - `outputs/air56_actor_ep001_loadgate_20260411b`
    - best partial row after `3/7` processed checkpoints:
      - `actor_ep002 + eta_mid_60_sp`
      - `avg_power_saving_pct = 1.0161607550775114`
      - `avg_eta_gain_pct = 0.12701110643196345`
      - `avg_power_saving_pct_min_seed = 0.8806894996602782`
      - `avg_eta_gain_pct_min_seed = 0.11647485611738229`
      - `load_step pass_count = 2/5`
    - conclusion:
      - tighter training-time `speed_tol` and `id_ref` gate did improve the shape of the tradeoff versus the previous dead branch
      - but after the first three checkpoints there was still no row-level lift beyond `load_step = 2/5`
      - this is still not enough to justify calling `AIR56` closed
  - explicit candidate-layer closure around the best new load-gate checkpoint is now also exhausted:
    - run:
      - `outputs/air56_ep002_loadgate_localsearch_20260411c_sync/air56_tuning_summary.json`
    - checkpoint under test:
      - `outputs/air56_actor_ep001_loadgate_20260411b/results_run/20260411_004825_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep002.pth`
    - result:
      - best remained baseline `eta_mid_60_sp`
      - exact metrics:
        - `avg_power_saving_pct = 1.0161607550775114`
        - `avg_eta_gain_pct = 0.12701110643196345`
        - `avg_power_saving_pct_min_seed = 0.8806894996602782`
        - `avg_eta_gain_pct_min_seed = 0.11647485611738229`
        - `load_step pass_count = 2/5`
      - all targeted softer load-gate variants remained below this baseline and still failed strict envelope acceptance
    - conclusion:
      - the new load-gate basin produced a better tradeoff than the previous reward-only dead branch
      - but its candidate layer is already exhausted at the current partial-best checkpoint
      - the only cheap remaining step on this branch is to finish the interrupted external checkpoint scan for the remaining `actor_ep003+`, `actor_ep004+`, `actor_ep005+`, and `actor_ep_init`
      - this branch was stopped to save compute on the weak machine
  - latest training-level load-step-heavy continuation with a materially stronger tracking objective also completed without closure:
    - `outputs/air56_ep002_loadheavy_wspeed2_20260408h/results_run/20260408_203735_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - training changed:
      - `w_speed = 2.0`
      - shorter continuation from `actor_ep002`
      - same strict multi-tag external selector
    - best result:
      - `actor_ep001 + gatepush_base`
      - `avg_power_saving_pct = 0.9939138562946258`
      - `avg_eta_gain_pct = 0.11796196074999099`
      - `avg_power_saving_pct_min_seed = 0.8902879268877617`
      - `avg_eta_gain_pct_min_seed = 0.07561588804608499`
      - `err_failures = 0.6`
      - `load_step pass_count = 2/5`
    - conclusion:
      - stronger tracking weight improved the branch relative to the failed `w_speed=1.0` variant
      - but it still did not move `AIR56` beyond the same `load_step = 2/5` barrier
      - therefore the cheap/medium continuation layer around the current `actor_ep002` frontier is now exhausted
  - important `2026-04-11` correction to the research record:
    - old `ai_id_ref` scenario-based AIR56 conclusions must be treated carefully because the train path had a real bug:
      - `train_ai_id_ref.build_env()` was still forcing constant `omega_ref_func` / `load_torque_func`
      - so runs launched with `--scenarios ...` were not actually training on those scenario functions
    - the same pass also exposed a second validity issue:
      - many old short AIR56 branches used `episode_steps = 150`
      - for `AIR56` with `dt = 5e-4` and `t_end = 2.0`, that horizon is only `0.075 s`
      - it does not even reach the canonical scenario events:
        - `speed_step = 400 steps`
        - `start_stop = 800 steps`
        - `load_step = 1200 steps`
        - `ramp = 2400 steps`
    - current execution rule:
      - do not use pre-fix short-horizon `ai_id_ref` scenario runs as strong negative evidence against scenario-aware AIR56 training
  - first corrected full-horizon scenario-valid baseline continuation is now completed:
    - run:
      - `outputs/air56_actor_ep001_true_loadstep_base_20260411g/results_run/20260411_031456_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - training recipe:
      - `episodes = 4`
      - `episode_steps = 2400`
      - scenarios cycled through `load_step,speed_step,start_stop,load_step`
      - no per-scenario reward overrides yet
    - result:
      - new valid selected checkpoint appeared instead of simply reselecting the warm-start incumbent:
        - `actor_ep003 + eta_mid_60_sp`
      - exact metrics:
        - `avg_power_saving_pct = 1.0290197779245958`
        - `avg_eta_gain_pct = 0.11979755825249683`
        - `avg_power_saving_pct_min_seed = 0.8996974381452388`
        - `avg_eta_gain_pct_min_seed = 0.09611936947062083`
        - `load_step pass_count = 2/5`
    - conclusion:
      - corrected scenario-aware training is now proven capable of producing a new AIR56 checkpoint
      - but canonical `load_step` closure still did not move past `2/5`
  - candidate-layer around that corrected scenario-valid checkpoint is already exhausted:
    - run:
      - `outputs/air56_actor_ep003_true_loadstep_localsearch_20260411h/air56_tuning_summary.json`
    - result:
      - best remained the checkpoint baseline `eta_mid_60_sp`
      - no targeted candidate lifted `load_step` above `2/5`
  - train/eval window-alignment branch with new `reward_start_frac` support is also now completed:
    - code path:
      - `mic_ai/ai/train_ai_id_ref.py`
      - per-scenario overrides now also support:
        - `reward_start_frac`
    - train run:
      - `outputs/air56_actor_ep003_rewardwindow_20260411i`
    - strict posthoc rescan:
      - `outputs/air56_actor_ep003_rewardwindow_20260411i/strict_rescan/air56_checkpoint_scan_summary.json`
    - result:
      - best strict row:
        - `actor_ep001 + gatepush_base`
      - exact metrics:
        - `avg_power_saving_pct = 1.0788537208274063`
        - `avg_eta_gain_pct = 0.11715761270042757`
        - `avg_power_saving_pct_min_seed = 0.9734497591340208`
        - `avg_eta_gain_pct_min_seed = 0.09295894213616207`
        - `load_step pass_count = 2/5`
      - new useful checkpoint inside this branch:
        - `actor_ep002 + rand007_soft_track`
        - `avg_power_saving_pct = 1.0667422514715807`
        - `avg_eta_gain_pct = 0.12542954320108546`
        - `avg_power_saving_pct_min_seed = 0.9529039434222009`
        - `avg_eta_gain_pct_min_seed = 0.11018576385412038`
        - `load_step pass_count = 2/5`
        - `load_step power_saving_pct_min = -0.07395594720649434`
    - conclusion:
      - late-window reward alignment alone does not close AIR56
      - but it does produce a new checkpoint with a better `load_step` floor than the corrected-scenario baseline
  - targeted local-safe retune around the new reward-window checkpoint is now exhausted:
    - run:
      - `outputs/air56_actor_ep002_rewardwindow_localsearch_20260411j/air56_tuning_summary.json`
    - checkpoint under test:
      - `outputs/air56_actor_ep003_rewardwindow_20260411i/results_run/20260411_035506_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep002.pth`
    - result:
      - best remained baseline `rw_ep002_rand007_base`
      - exact metrics:
        - `avg_power_saving_pct = 1.0667422514715807`
        - `avg_eta_gain_pct = 0.12542954320108546`
        - `avg_power_saving_pct_min_seed = 0.9529039434222009`
        - `avg_eta_gain_pct_min_seed = 0.11018576385412038`
        - `load_step pass_count = 2/5`
    - conclusion:
      - the candidate-layer on top of the new reward-window checkpoint is already exhausted
      - next justified AIR56 step is again training-level, not another local-search recycle
  - reward-window branch with stronger in-training load-step guardrail shaping is now also closed as a dead end:
    - run:
      - `outputs/air56_actor_ep002_rewardwindow_guard_20260411k/results_run/20260411_044817_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - training changed:
      - warm-start from the new reward-window checkpoint `actor_ep002`
      - kept full horizon `2400`
      - added `load_step` overrides:
        - `reward_start_frac = 0.55`
        - `w_speed = 3.0`
        - `ai_id_speed_tol = 0.40`
        - `id_ref_gate_speed_tol_rel = 0.10`
        - `id_ref_gate_min_scale = 0.05`
        - `id_ref_gate_exponent = 1.35`
      - built-in external selector was already run in canonical mode (`--external-step27-use-envelope-acceptance`)
    - result:
      - selector re-promoted the warm-start init checkpoint:
        - `actor_ep_init + rand007_soft_track`
      - exact metrics stayed identical to the previous reward-window frontier:
        - `avg_power_saving_pct = 1.0667422514715807`
        - `avg_eta_gain_pct = 0.12542954320108546`
        - `avg_power_saving_pct_min_seed = 0.9529039434222009`
        - `avg_eta_gain_pct_min_seed = 0.11018576385412038`
        - `load_step pass_count = 2/5`
    - conclusion:
      - stronger in-training load-step guardrail shaping did not improve the frontier
      - this reward-window checkpoint lineage is now exhausted at both candidate and short continuation levels
  - corrected scenario-aware continuation from the strict incumbent is also now closed as a dead end:
    - run:
      - `outputs/air56_incumbent_etawindow_20260411l/results_run/20260411_051324_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - training changed:
      - warm-start from strict incumbent `actor_ep005 + mix04_base`
      - full horizon `2400`
      - candidate set expanded to:
        - `mix04_base`
        - `rand007_soft_track`
        - `eta_mid_60_sp`
      - `load_step` late-window eta-focused override:
        - `reward_start_frac = 0.55`
        - `w_eta = 1.5`
        - `w_eta_episode = 0.2`
    - result:
      - selector again re-promoted `actor_ep_init + mix04_base`
      - exact metrics remained the old incumbent baseline:
        - `avg_power_saving_pct = 0.6147884835796003`
        - `avg_eta_gain_pct = 0.0013968558560217836`
        - `avg_power_saving_pct_min_seed = 0.5151508570179902`
        - `avg_eta_gain_pct_min_seed = -0.019478794161115198`
        - `envelope_all_rows_pass = true`
    - conclusion:
      - after the scenario-path fix, short eta-window continuation from the strict incumbent still does not move the negative eta tail
      - the strict incumbent lineage is now also exhausted for cheap corrected scenario-aware continuation
  - historical `speedfix_ft actor_ep015` basin is also now ruled out for the current strict objective:
    - `outputs/air56_speedfix_actor015_candidategrid_20260326/air56_tuning_summary.json`
    - best candidate remained `mix04_base`
    - result:
      - `avg_power_saving_pct_min_seed = 0.12568332259845771`
  - reserve-path checks already ruled out:
    - `ai_current` strict rescan is dead:
      - `outputs/air56_aicurrent_reltrack_20260328a/results_run/20260328_104205_tmp_air56_refine_ep002_pin60hb_20260322_ai_current/external_step27_rescan_strict/air56_checkpoint_scan_summary.json`
      - `err_failures = 4.0`
      - `envelope_all_rows_pass = false`
    - `ai_voltage` 1-seed smoke is dead:
      - `outputs/air56_aivoltage_smoke_20260328a/scan_1seed/air56_checkpoint_scan_summary.json`
      - `err_failures = 4.0`
      - `worst_current_peak_ratio = 4.47059281007757`
      - `envelope_all_rows_pass = false`
    - `ai_speed` semantic-fix rerun is dead even after correct action-head remap:
      - `outputs/air56_aispeed_semanticfix_20260328d/results_run/20260328_121212_env_research_air56_025kw_ai_speed/external_step27_scan/air56_checkpoint_scan_summary.json`
      - result:
        - all `12/12` checkpoints remained at `avg_power_saving_pct = 100.0`
        - `avg_eta_gain_pct = -75.0`
        - `err_failures = 4.0`
        - `envelope_all_rows_pass = false`
      - conclusion:
        - the old `ai_speed` failure was not just a warm-start action-slot bug
        - `ai_speed` is not a viable near-pass reserve for current AIR56 W1
    - `foc_assist` semantic-fix rerun improved the early frontier but still remained far red:
      - `outputs/air56_focassist_semanticfix_20260328b/results_run/20260328_122027_tmp_air56_focassist_energy_smoke_20260328_foc_assist/external_step27_scan/air56_checkpoint_scan_summary.json`
      - best checkpoint:
        - `actor_ep002`
      - result:
        - `avg_power_saving_pct = 62.59316457355009`
        - `avg_eta_gain_pct = -22.959951640003034`
        - `err_failures = 3.0`
        - `envelope_all_rows_pass = false`
      - conclusion:
        - semantic remap mattered, but only enough to make `foc_assist` "less bad"
        - it is still nowhere near strict closure and is not a justified next medium-budget branch
  - current AIR56 conclusion after the `2026-04-11` corrections:
    - the active frontier is no longer the old reltrack-only story
    - there are now three relevant AIR56 states:
      - strict incumbent:
        - envelope-clean
        - eta-tail red
      - corrected scenario-valid checkpoint `actor_ep003 + eta_mid_60_sp`:
        - proves scenario-aware training now works after the train-path fix
        - still `load_step = 2/5`
      - reward-window checkpoint `actor_ep002 + rand007_soft_track`:
        - currently the best corrected-train frontier for `load_step power_saving_pct_min`
        - still `load_step = 2/5`
    - what is now exhausted:
      - candidate-layer around the corrected scenario-valid checkpoint
      - candidate-layer around the reward-window checkpoint
      - short continuation from the reward-window checkpoint with stronger in-training `load_step` guardrails
      - short corrected scenario-aware eta-window continuation from the strict incumbent
    - next AIR56 step must therefore be materially different:
      - either capacity-shift continuation from the strict incumbent on the corrected scenario path
      - or corrected full-horizon scratch / rebuild on the `AIR56` recipe
      - keep full `2400`-step horizon
      - keep full `5-seed` strict external selection
    - do not spend more compute on:
      - another candidate-only search around `actor_ep003`
      - another candidate-only search around `actor_ep002`
      - another short continuation on `actor_ep002` with the same `64x64/128x128` reward-window recipe
      - another short incumbent eta-window continuation with the same `64x64` hidden-size regime
      - another short-horizon scenario run

#### W1.2 New code status
- reward alignment change added in `mic_ai/ai/ai_env.py`:
  - new helper: `compute_energy_reward_gate(...)`
  - new helper: `compute_running_eta_penalty(...)`
  - new config knobs:
    - `ai_id_energy_gate_mode`
    - `ai_id_energy_gate_min_scale`
    - `ai_id_energy_gate_exponent`
    - `w_ai_id_eta_episode`
    - `ai_id_terminal_energy_bonus`
    - `ai_id_terminal_eta_target`
    - `ai_id_terminal_shaft_ratio_min`
  - new observation key available for experiments:
    - `eta_episode_norm`
  - default mode remains `hard`, so old behavior is preserved unless explicitly enabled
- regression added:
  - `tests/test_ai_env_reward_gate.py`
  - `tests/test_ai_env.py`
  - `tests/test_train_ai_id_ref_external_step27.py`
- factual status:
  - code is implemented locally and tested
  - unit/regression tests are green
  - important trainer/eval bugs were fixed:
    - `train_ai_id_ref.build_env()` no longer overwrites scenario-defined `omega_ref_func` / `load_torque_func` when `--scenarios` is used
    - `train_ai_id_ref` now warns when `episode_steps` is shorter than the selected scenario activation horizon
    - `train_ai_id_ref` now supports per-scenario `reward_start_frac` so training reward can be aligned to the late evaluation window instead of rewarding startup transients
    - `train_ai_id_ref.build_env()` now really propagates reward-gate / terminal-bonus knobs from env-config into `AiEnvConfig`
    - warm-start loading now supports appended observation features by zero-padding new input columns
    - warm-start loading now also supports wider hidden layers by zero-padding overlapping slices into expanded tensors
    - `train_ai_id_ref` can now infer hidden sizes from a warm-start checkpoint instead of silently building the wrong width
    - `train_ai_id_ref` can now apply reward overrides from CLI, which avoids the `from ... import *` / `__all__ = ["ENV"]` trap in temporary configs
    - `train_ai_id_ref` now supports energy curriculum and explicit `--hidden-sizes`
    - external Step27 scan now accepts explicit `feature_keys` and can evaluate mixed feature-dimension checkpoint lineages
    - `scan_step27_checkpoints.py` now writes incremental `*_checkpoint_scan_progress.json` during long scans so strict runs are observable before final summary
    - `scan_step27_checkpoints.py` now writes persistent `*_checkpoint_scan_state.json` and supports `--resume` after interrupted scans
    - `scan_step27_checkpoints.py` can now rank each checkpoint across a tiny shortlist of `ai_id_ref` candidates (`--candidate-tags`) instead of forcing one fixed candidate for the whole scan
    - `train_ai_id_ref.py` now exposes `--external-step27-resume` so interrupted built-in external selection can be resumed without rerunning training
    - `train_ai_id_ref.py` now forwards `--external-step27-candidate-tags` into built-in external Step27 checkpoint selection
    - `train_ai_id_ref.py` now infers the feature-space from a warm-start checkpoint when the current default feature set is only a strict prefix of the checkpoint feature set
    - `train_ai_id_ref.py` now forwards the real `control_mode` into external Step27 checkpoint selection, so `ai_current` reserve-paths can be evaluated truthfully instead of silently being scanned as `ai_id_ref`
    - `train_ai_id_ref.py` and external loaders now semantically remap a legacy single-action `id_ref` checkpoint into the second action slot when crossing into `ai_current` / `ai_speed` / `foc_assist`
    - `mic_ai/tools/scenario_compare.py::_infer_action_dim()` now infers action dimension from both legacy and current PPO head layouts (`actor_mu.*`, `actor_head.*`, `log_std`)
    - `mic_ai/tools/scenario_compare.py`, `tools/step27_pipeline.py`, and `tools/scan_step27_checkpoints.py` now accept `foc_assist` / `ai_speed` control modes in their CLIs
    - `tools/step27_pipeline.py::_load_agent()` now adapts 1D and 2D checkpoint tensors generically, so mixed action-head / feature-layout checkpoints can be loaded in external Step27 scan without crashing
    - `tools/step27_pipeline.py::_load_agent()` now falls back to checkpoint-inferred feature keys when explicitly supplied feature keys do not match the checkpoint feature layout
    - `train_ai_id_ref.build_env()` now enables two-action `id` control for `foc_assist` / `ai_speed`, which matches the actual `AiEnv` implementations
    - `train_ai_id_ref` no longer forces `--external-step27-candidate-json` for non-`ai_id_ref` modes
    - `ai_env.py` / `metrics.py` now clamp the reserve-mode numeric hotspots that previously produced overflow in core-loss, dq/abc current RMS, mechanical power, and raw float32 observation casts
  - real validation after those fixes:
    - `AL31` true soft-gate rerun still kept `actor_ep_init` as best:
      - `outputs/al31_mid04_softgate_fixed_20260322/results_run/20260322_223527_tmp_al31_mid04_train_softgate_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - `AL31` terminal-bonus rerun still kept `actor_ep_init` as best:
      - `outputs/al31_mid04_terminal_fixed_20260322/results_run/20260322_224103_tmp_al31_mid04_train_terminal_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - `AIR56` terminal-bonus rerun still kept `actor_ep_init` as best:
      - `outputs/air56_ep005_terminal_fixed_20260322/results_run/20260322_224909_tmp_air56_ep022_mix04_train_terminal_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - `AIR56` running-eta reward plus explicit `eta_episode_norm` observation still kept `actor_ep_init` as best:
      - `outputs/air56_ep005_runningeta_obs_micro1_fix_20260322/results_run/20260322_230543_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - first materially different scratch recipe on `AL31` did change the selected checkpoint, but regressed envelope/start-stop and still failed eta:
      - `outputs/al31_basinshift_scratch1_20260322/results_run/20260322_233249_tmp_al31_basinshift_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - first materially different scratch recipe on `AIR56` collapsed below the current baseline on power, eta, and `start_stop`:
      - `outputs/air56_basinshift_scratch1_20260323/results_run/20260322_234518_tmp_air56_basinshift_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - valid two-stage `64x64` base-config run on `AL31` improved mean energy but still lost `start_stop` / min-seed strict closure:
      - `outputs/al31_twostage_valid_basecfg_20260323/results_run/20260323_005211_env_research_al31_4_06kw_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - valid two-stage `64x64` base-config run on `AIR56` kept the old baseline best:
      - `outputs/air56_twostage_valid_basecfg_20260323/results_run/20260323_012647_env_research_air56_025kw_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - first valid capacity-shift run on `AL31` (`96x96`) produced a new strict best checkpoint `actor_ep026`, but still left a tiny negative min-seed eta tail:
      - `outputs/al31_capacity96_valid_basecfg_20260323/results_run/20260323_014139_env_research_al31_4_06kw_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - local-safe tuning around `AL31 actor_ep026` did not close that remaining tail:
      - `outputs/al31_ep026_capacity96_localsearch_20260323/al31_tuning_summary.json`
    - `AIR56` relaxed Phase A detached rescan finished and re-selected the old baseline `actor_ep_init`:
      - `outputs/air56_phaseA_powerguard_20260323a/results_run/20260323_104558_env_research_air56_025kw_ai_id_ref/external_step27_rescan/air56_checkpoint_scan_summary.json`
    - `AIR56` actor-anchor eta micro-run finished and selected `actor_ep000`, not the warm-start baseline and not a passing checkpoint:
      - `outputs/air56_anchor_eta_micro1_20260323a/20260323_123204_env_research_air56_025kw_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
  - meaning:
    - reward and feature extensions are now scientifically validated, not just implemented
    - they still do not beat the current strict baselines under low-compute micro-finetune conditions
    - therefore the remaining blocker is no longer "missing config plumbing" or "missing local gradient"
    - the blocker is the current PPO local basin / compute budget itself
    - a naive scratch basin-shift is also not sufficient: it reopens tracking/start-stop regressions before closing worst-case eta
    - a relaxed Phase A curriculum on `AIR56` also did not move the selected checkpoint away from the old baseline
    - a first actor-anchor objective on `AIR56` also failed to close the strict eta tail
    - the current active question is no longer actor-anchor on `AL31`
    - the current active question is whether a larger `AIR56` reltrack continuation can turn the new frontier into a fully strict checkpoint without reopening the old tails now that the mixed-feature bug is fixed

#### W1.3 Immediate execution order
1. Stop reopening the current `AL31` and `AIR56` local basins with more micro-runs that only tweak reward scalars or one extra observation.
2. Keep the current factual baselines:
   - strict `AL31`: `actor_ep008 + al31_mid_04`
   - strict `AIR56` incumbent: `actor_ep005 + mix04_base`
   - `AIR56` near-pass frontier: `actor_ep004 + rand007_soft_track`
3. Next justified W1 step is now materially different from all exhausted low-compute runs:
   - keep strict external Step27 selector
   - change the checkpoint frontier, not just small scalar weights
   - acceptable next classes:
     - `AIR56`: medium-budget continuation from the reltrack lineage with external selection over the `mix04_rand_007` / `mix04_rand_015` frontier
     - `AL31`: medium-budget checkpoint run from `actor_ep008` with strict min-seed eta selector
     - constrained two-stage curriculum: first preserve/recover envelope and `start_stop`, then push energy
     - longer scratch training only if it includes explicit tracking/start-stop protection, not the naive scratch recipe already tested
     - different hidden-size / policy-capacity regime for the MIC `id_ref` policy
     - if needed, a different optimizer / algorithmic family, not another micro-PPO replay of the same basin
4. Immediate next active experiment:
   - `AIR56` cross-lineage strict replay under one canonical candidate was started and partially consumed:
     - run root: `outputs/air56_crosslineage_mix04_strict_20260326g_pruned`
     - candidate:
       - `outputs/tmp_air56_mix04_single_candidate_20260322.json`
     - purpose:
       - test whether a hidden strict winner already exists inside the accumulated `AIR56` checkpoint lineages
       - avoid another expensive train-cycle if the answer is already on disk
     - setup:
       - pruned shortlist of `28` frontier checkpoints copied from historical `AIR56` scan summaries
       - strict external Step27 selector on `5` seeds with canonical envelope acceptance
       - same strict thresholds as the current incumbent branch
     - stopping rationale:
       - after the first consumed frontier block, the best replayed checkpoint was already another incumbent-like clone
       - remaining shortlist rows were only weaker historical basins or exact incumbent clones
       - continuing the replay was no longer compute-efficient on the weak machine
  - completed on `2026-03-26`:
    - `AIR56` reltrack-speedfix hybrid continuation
    - run root: `outputs/air56_reltrack_speedfix_hybrid_20260326h`
    - warm-start checkpoint:
      - `outputs/air56_ep009_reltrack_train_20260322/results_run/20260322_091834_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/eval/actor_ep004.pth`
    - preserved deployment candidate during selection:
      - `outputs/tmp_air56_rand007_soft_track_single_20260326.json`
    - result:
      - strict selector again re-selected `actor_ep_init`
      - artifact:
        - `outputs/air56_reltrack_speedfix_hybrid_20260326h/results_run/20260326_144414_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
      - best row:
        - `avg_power_saving_pct = 0.33362384869261963`
        - `avg_eta_gain_pct = 0.009031393618700867`
        - `avg_power_saving_pct_min_seed = 0.17977661337638395`
        - `avg_eta_gain_pct_min_seed = -0.001159630724775762`
        - `err_failures = 0.2`
        - `envelope_all_rows_pass = false`
      - conclusion:
      - this branch confirmed the reltrack near-pass frontier but still left exactly one `speed_step` fail
      - it did not produce a new strict-closing checkpoint
  - reserve paths now empirically narrowed:
    - `ai_current` can now be evaluated truthfully because external Step27 selection no longer loses the MIC control mode during scan
    - that path is now tested and ruled out for current `AIR56`
    - `ai_voltage` support is also usable in principle, but its first feasibility probe on `AIR56` is already red enough that it should not be promoted to a medium/full-budget closure run
  - active as of `2026-03-26` now:
    - completed on `2026-03-28`:
      - `AIR56` reltrack `96x96` anchored continuation
      - run root: `outputs/air56_reltrack_capacity96_anchor_20260326i`
      - result:
        - strict selector again re-selected `actor_ep_init`
        - artifact:
          - `outputs/air56_reltrack_capacity96_anchor_20260326i/results_run/20260326_150147_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
        - conclusion:
          - the larger `96x96` capacity regime did not beat the existing reltrack frontier
          - this closes the current capacity-hypothesis branch for the reltrack lineage
    - completed on `2026-03-28`:
      - `AIR56` `ai_current` warm-start from the reltrack frontier
      - run root: `outputs/air56_aicurrent_reltrack_20260328a`
      - warm-start checkpoint:
        - `outputs/air56_reltrack_speedfix_hybrid_20260326h/results_run/20260326_144414_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/eval/actor_ep_init.pth`
      - preserved deployment candidate during selection:
        - `outputs/air56_base_current_candidate_20260318.json`
      - setup:
        - control regime is changed from `ai_id_ref` to `ai_current`
        - external Step27 selector now receives the truthful `ai_control_mode`
        - purpose:
          - test a genuinely different MIC action family instead of another reltrack `ai_id_ref` replay
      - strict rescan artifact after loader/action-dim fix:
        - `outputs/air56_aicurrent_reltrack_20260328a/results_run/20260328_104205_tmp_air56_refine_ep002_pin60hb_20260322_ai_current/external_step27_rescan_strict/air56_checkpoint_scan_summary.json`
      - result:
        - best checkpoint `actor_ep005.pth`
        - `avg_power_saving_pct = 76.78497663961898`
        - `avg_eta_gain_pct = 43687.82130096828`
        - `err_failures = 4.0`
        - `avg_eta_gain_pct_min_seed = -75.0`
        - `envelope_all_rows_pass = false`
        - `envelope_fail_count = 20`
      - conclusion:
        - `ai_current` is not a near-pass reserve for `AIR56`
        - it is qualitatively unstable and far worse than both incumbent and reltrack `ai_id_ref` frontiers
        - do not spend more compute on `AIR56 ai_current` within W1
    - completed on `2026-03-28`:
      - `AIR56` `ai_voltage` feasibility smoke
      - train root: `results_run/20260328_110134_env_research_air56_025kw`
      - 1-seed strict probe:
        - `outputs/air56_aivoltage_smoke_20260328a/scan_1seed/air56_checkpoint_scan_summary.json`
      - result:
        - `avg_power_saving_pct = 83.68920347893385`
        - `avg_eta_gain_pct = -275.5657243391105`
        - `err_failures = 4.0`
        - `worst_current_peak_ratio = 4.47059281007757`
        - `envelope_all_rows_pass = false`
      - conclusion:
        - this family is dead-on-arrival for current `AIR56`
        - do not escalate `ai_voltage` to a medium/full-budget branch until there is a new control objective / constraint recipe
  - next class after the failed `ai_current` and `ai_voltage` reserve probes:
    - switch to a more materially different AIR56 basin / optimizer / curriculum change
    - rationale:
    - `AIR56` remains the larger blocker
    - the incumbent-side continuation is now proven unable to move the tail
    - local candidate search is already proven exhausted on both fronts
    - the historical speedfix basin is now also proven not to be the closure path
    - the reltrack hybrid branch is now also proven unable to close the row at `64x64`
    - the reltrack `96x96` capacity branch is now also proven unable to close the row
    - `ai_current` is now also proven not to be the closure path
    - `ai_voltage` feasibility is also red enough to reject as the next cheap reserve
    - the next prepared AIR56 basin is now explicit, not abstract:
      - primary next candidate: `foc_assist`
      - fallback next candidate: `ai_speed`
      - both are now first-class in train/scan/eval and do not require a new evaluation stack
  - completed on `2026-03-28`:
    - `AL31` medium-budget checkpoint run from `actor_ep008`
    - run root: `outputs/al31_anchor_ep008_medium4_20260328a`
    - result:
      - strict selector re-selected the warm-start incumbent
      - `avg_eta_gain_pct_min_seed` remained `-0.00034895129392975566`
      - `envelope_all_rows_pass = true`
    - conclusion:
      - `AL31 ai_id_ref` medium branch is exhausted for now
      - keep `AL31` parked as a tiny-tail blocker while AIR56 remains the dominant unresolved motor
  - current AIR56 execution conclusion:
    - all prepared reserve control families are now checked on cleaner evidence:
      - `ai_current`: dead
      - `ai_voltage`: dead
      - `ai_speed`: dead even after semantic warm-start fix
      - `foc_assist`: improved after semantic fix, but still far red
    - the eta-frontier short continuation is also now checked and dead as a source of new winners
    - therefore the next justified AIR56 step is no longer another reserve control family and no longer another cheap frontier continuation
    - it must be a deeper basin/objective shift in the `ai_id_ref` line itself, or a genuinely different optimizer/policy regime
5. Only after a materially different recipe closes both remaining tails, rerun full 3-motor live Step27 and then rebuild Step28 candidate.

#### W1.4 Acceptance for W1 closure
- `AIR56` remains green in the same final strict `p0.2` full run.
- `AL31` remains green in the same final strict `p0.2` full run.
- `AO2` strict `p0.2` pass:
  - `avg_power_saving_pct >= 0`
  - `avg_eta_gain_pct >= 0`
  - `avg_power_saving_pct_min_seed >= 0`
  - `avg_eta_gain_pct_min_seed >= 0`
  - `err_failures <= 2`
  - `start_stop_power_saving_pct >= -0.5`
  - `start_stop_power_saving_pct_min_seed >= -0.5`
  - `worst_current_peak_ratio <= 1.30`
  - `worst_current_mean_ratio <= 1.20`
  - `envelope_all_rows_pass = true`

### W2. Step28 reproduce / verify / freeze
Goal: convert W1 green state into a real submission-ready candidate.

Execute only after W1 is green.

Steps:
1. Rebuild the canonical live Step27 run for all 3 motors.
2. Reproduce Step28 in `--mic-mode ai` only.
3. Rebuild passport block if needed.
4. Rebuild dossier / checklist / verify artifacts.
5. Freeze the new candidate only if `verification_ok=true`.

Acceptance:
- new candidate under `paper/ieee_2026/data/step28/...`
- `FINAL_CHECKLIST_AUTO.md` green
- `VERIFY_SUBMISSION_CANDIDATE.json` green

### W3. Universal any-motor closure
Goal: finish the generic algorithm track, not just the 3-motor restore branch.

Current status:
- correctness gate is already green when revalidating from an existing checkpoint
- energy gate is still red on the verification set
- identification-first proof is missing

Remaining:
1. close energy gate on the designated benchmark set
2. add documented passport-only flow artifact
3. add documented passport + identification flow artifact
4. prove the flow on a real identification-first scenario

Acceptance:
- `all_ok=true` on correctness and energy reports
- docs and example artifacts exist for both flows

### W4. Refactor orchestration layer
Goal: reduce script size and duplicated responsibilities.

Targets:
- `tools/step27_pipeline.py`
- `tools/train_any_motor_pipeline.py`
- `tools/train_3motors_pipeline.py`
- `tools/robust_motor_hardening.py`

Remaining:
- extract shared checkpoint-scan / acceptance / report logic
- separate training orchestration from packaging/report generation
- remove duplicate gate calculations where possible

Acceptance:
- smaller modules with clearer ownership
- no behavior regression in focused tests

### W5. Documentation and repository hygiene
Goal: make the repo readable and consistent.

Remaining:
- normalize `README.md` to UTF-8 and rewrite outdated sections
- update runbooks to match current entrypoints and gate semantics
- keep only one active plan in root
- avoid leaving ambiguous duplicate instructions in root

Acceptance:
- `README.md` reads correctly in shell/editor
- docs reflect real current flows

### W6. Final test and regression hardening
Goal: protect the final closed state.

Remaining:
- add smoke/regression for `--skip-training --init-checkpoint`
- add tests for benchmark search ranking / gate pass-fail modes
- add root hygiene regression
- keep reward-gate regression tests green
- run final `pytest -q` after cleanup/refactor

Acceptance:
- `pytest -q` green after all final cleanup

## Current phased execution plan
This is the strict execution sequence from the current state. Work must move phase by phase without parallel detours.

### Phase 1. Close `AIR56` row-level blocker
1. Freeze the factual baselines and do not relitigate them:
   - `AL31`: `actor_ep_init + mid04_speed_dn_04`
   - `AIR56` strict incumbent: `actor_ep005 + mix04_base`
   - `AIR56` strongest corrected scenario-valid frontier:
     - checkpoint: `actor_ep002` from `outputs/air56_actor_ep003_rewardwindow_20260411i/results_run/20260411_035506_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep002.pth`
     - strongest strict cross-family deploy pair after the `2026-04-11` scenario-path fix:
       - `actor_ep002 + rand007_soft_track`
       - artifact: `outputs/air56_actor_ep003_rewardwindow_20260411i/strict_rescan/air56_checkpoint_scan_summary.json`
       - exact metrics:
         - `avg_power_saving_pct = 1.0667422514715807`
         - `avg_eta_gain_pct = 0.12542954320108546`
         - `avg_power_saving_pct_min_seed = 0.9529039434222009`
         - `avg_eta_gain_pct_min_seed = 0.11018576385412038`
         - `load_step pass_count = 2/5`
         - `envelope_all_rows_pass = false`
2. Treat the current `AIR56` blocker as exactly one problem:
   - aggregate thresholds are already green on the latest frontier
   - the blocker is now broader than the old simplified description "`load_step` only":
     - the best current cheap cross-family pair still fails canonical envelope at `load_step = 2/5`
     - archival pairs that looked green on relaxed scans do not survive a fresh strict `p0.2` recheck
   - strict rechecks completed on `2026-04-10` and already ruled out as final AIR56 proofs:
     - `actor_ep005 + rand007_soft_track`
       - `outputs/air56_actor_ep005_rand007_strictp02_recheck_20260410a/air56_checkpoint_scan_summary.json`
       - failed `speed_step` row and strict min-seed aggregate thresholds
     - `best_actor + eta_mid_60_sp`
       - `outputs/air56_bestactor_eta60sp_strictp02_recheck_20260410a/air56_checkpoint_scan_summary.json`
       - failed `speed_step` row and strict aggregate thresholds
3. Do not spend more compute on already exhausted `AIR56` branches:
   - candidate-only retunes around `ep002/ep003/ep006/ep019`
   - reserve control families `ai_current`, `ai_voltage`, `ai_speed`, `foc_assist`
   - near-identical short continuations that preserve the same local basin
   - the extra `2026-04-10` cheap dead ends are now also closed:
     - `outputs/air56_ep001_loadheavy_wspeed3_20260410a`
       - short training branch with stronger `w_speed=3.0`
       - shortlist checkpoints degraded to `load_step = 1/5`
     - `outputs/air56_loadstep_curriculum_20260410a`
       - short loadstep-curriculum branch with narrow omega/load randomization
       - shortlist checkpoints also degraded to `load_step = 1/5`
     - `outputs/air56_actor_ep001_rand007_localsearch_20260410a/air56_tuning_summary.json`
       - narrow local retune around the best current cross-family pair
       - best remained `rand007_base`; no envelope closure
     - `outputs/air56_actor_ep002_rewardwindow_localsearch_20260411j/air56_tuning_summary.json`
       - local-safe retune around the corrected scenario-valid `actor_ep002 + rand007_soft_track` frontier
       - best remained baseline; `load_step` stayed at `2/5`
     - `outputs/air56_actor_ep002_rewardwindow_guard_20260411k/results_run/20260411_044817_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
       - reward-window continuation plus stronger in-training `load_step` guard shaping
       - selector re-promoted the init checkpoint and did not improve strict closure
     - `outputs/air56_incumbent_etawindow_20260411l/results_run/20260411_051324_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
       - corrected short eta-window continuation from the strict incumbent
       - dead end; old strict incumbent remained best
     - `outputs/air56_incumbent_capacity128_etawindow_20260411m/results_run/20260411_053345_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
       - corrected `128x128` capacity-shift continuation from the strict incumbent
       - dead end; old strict incumbent remained best
     - `outputs/air56_scratch128_rewardwindow_20260411n/results_run/20260411_060906_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
       - corrected full-horizon scratch `128x128` reward-window branch
       - dead end; selected checkpoint stayed far below strict closure
     - `outputs/air56_incumbent_mix04_eta_localsearch_20260411o/air56_tuning_summary.json`
       - targeted runtime search around the strict incumbent
       - baseline `mix04_base` remained best; no eta-tail closure
     - `outputs/air56_ep002_balanced_strict_20260411p`
       - corrected balanced continuation from the `actor_ep002` frontier with strict envelope selector from the start
       - full strict replay on the produced checkpoints showed no closure:
         - best aggregate-strong row after six evaluated checkpoints:
           - `actor_ep000 + rand007_soft_track`
           - `avg_power_saving_pct = 1.1255825816269671`
           - `avg_eta_gain_pct = 0.1260625171975549`
           - `avg_power_saving_pct_min_seed = 0.9851686556544309`
           - `avg_eta_gain_pct_min_seed = 0.10374944329870628`
           - `load_step pass_count = 2/5`
         - strict init recheck in the same branch also only reproduced the old corrected frontier:
           - `actor_ep_init + rand007_soft_track`
           - `avg_power_saving_pct = 1.0667422514715807`
           - `avg_eta_gain_pct = 0.12542954320108546`
           - `load_step pass_count = 2/5`
       - conclusion:
         - the balanced corrected continuation did not move the canonical `load_step` blocker
         - cheap candidate-layer around its promising `actor_ep004` checkpoint also stayed red:
           - `outputs/air56_ep004_loadfix_directscan_20260411r/air56_checkpoint_scan_summary.json`
           - best candidate became `ep004_track_up_03`, but `load_step pass_count` remained `2/5`
     - `outputs/air56_ep000_trackclosure_medium_20260411s`
       - medium tracking-first continuation from the strongest produced balanced checkpoint `actor_ep000`
       - shortlist strict replay over `actor_ep000, ep004, ep006, ep007, init` still re-selected the old corrected frontier:
         - `actor_ep_init + rand007_soft_track`
         - artifact: `outputs/air56_ep000_trackclosure_medium_20260411s/shortlist_scan_sync/air56_checkpoint_scan_summary.json`
         - exact metrics:
           - `avg_power_saving_pct = 1.1255825816269671`
           - `avg_eta_gain_pct = 0.1260625171975549`
           - `avg_power_saving_pct_min_seed = 0.9851686556544309`
           - `avg_eta_gain_pct_min_seed = 0.10374944329870628`
           - `load_step pass_count = 2/5`
       - conclusion:
         - medium continuation on the current corrected basin is also exhausted
         - the next justified move is no longer another continuation, but a new corrected scratch / rebuild recipe
4. Next `AIR56` run must change the training regime itself while staying in `ai_id_ref`:
   - objective or curriculum must target joint `load_step + speed_step` robustness instead of overfitting one scenario
   - external selection stays strict and canonical
   - deployment ranking stays against known real candidates, not synthetic thresholds only
   - active branch must now be a new corrected scratch / rebuild:
     - no warm-start
     - tracking-first scenario mix with repeated `load_step` / `speed_step`
     - lower exploration than the earlier scratch branch
     - delayed power / energy ramps so the policy first learns the canonical tracking envelope
     - strict selector remains the same:
       - `rand007_soft_track`
       - `gatepush_base`
       - `eta_mid_60_sp`
     - intended outcome:
       - find a new basin that can push canonical `load_step` above `2/5`
       - then only afterward recover/validate aggregate power and eta under the same strict gate
5. `AIR56` exit criterion:
   - `avg_power_saving_pct >= 0.5`
   - `avg_eta_gain_pct >= 0.0`
   - `avg_power_saving_pct_min_seed >= 0.5`
   - `avg_eta_gain_pct_min_seed >= 0.0`
   - `envelope_all_rows_pass = true`

### Phase 2. Close `AL31` tiny worst-case eta tail
1. Keep `AL31` parked until `AIR56` is green.
2. Do not reopen medium `ai_id_ref` continuations or dense local-safe runtime searches already shown exhausted.
3. After `AIR56` closure, run only one narrow `AL31` tail-closure branch aimed at:
   - keeping `envelope_all_rows_pass = true`
   - lifting `avg_eta_gain_pct_min_seed` from `-0.000206...` to `>= 0`
4. `AL31` exit criterion:
   - `avg_power_saving_pct >= 0`
   - `avg_eta_gain_pct >= 0`
   - `avg_power_saving_pct_min_seed >= 0`
   - `avg_eta_gain_pct_min_seed >= 0`
   - `envelope_all_rows_pass = true`

### Phase 3. Rebuild post-restore proof
1. Only after `AIR56` and `AL31` are both strict-green, rerun full live `Step27` for all three motors.
2. Rebuild `Step28` in `--mic-mode ai`.
3. Rebuild checklist, dossier, passport block, and verify artifacts.
4. Accept only a candidate with:
   - `ready_for_submission=true`
   - `checklist_ready_for_submission=true`
   - `verification_ok=true`

### Phase 4. Finish universal onboarding
1. Close the energy gate on the verification benchmark.
2. Produce and document both flows:
   - passport-only
   - passport + identification
3. Acceptance:
   - correctness green
   - energy green
   - real reproducible artifacts for both flows

### Phase 5. Finish engineering cleanup
1. Refactor monolithic orchestration scripts.
2. Normalize `README.md` and runbooks.
3. Run final full `pytest -q` on the cleaned repo.
4. Project is `100%` done only after all previous phases are green together.

