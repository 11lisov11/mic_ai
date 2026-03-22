# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-03-22`
Repository: `C:\mic_theory`

## Canonical source
- This file is the only active master plan allowed in the repository root.
- Historical root plans and root execution logs are archived under `docs/plan_archive/`.
- As of this refresh, the previous root files were moved to:
  - `docs/plan_archive/2026-03-16_plan_refresh/PROJECT_MASTER_PLAN_20260311_snapshot.md`
  - `docs/plan_archive/2026-03-16_plan_refresh/PROJECT_MASTER_EXECUTION_LOG_20260303_cycle2.md`

## Current factual baseline
- Stable working git root is now `C:\mic_theory`:
  - the workspace was consolidated back into `C:\mic_theory`
  - `C:\mt` and `C:\mic_theory_repo_restored` are now only duplicate/transition roots and must not be treated as canonical
- Test baseline is green:
  - `pytest -q` -> `117 passed` on `2026-03-16`.
- Focused restoration tests are green in the restored repo:
  - `pytest -q tests/test_tune_motor_step27_candidates.py tests/test_scan_step27_checkpoints.py tests/test_scenario_randomization.py tests/test_train_ai_id_ref_external_step27.py tests/test_train_ai_voltage.py`
  - `15 passed` on `2026-03-18`
- Focused selector/regression tests remain green after strict external-selection expansion:
  - `pytest -q tests/test_train_ai_id_ref_external_step27.py tests/test_scan_step27_checkpoints.py tests/test_tune_motor_step27_candidates.py`
  - `14 passed` on `2026-03-22`
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
- Restored repo reality on `2026-03-18`:
  - the restored root no longer contains any live motor checkpoints:
    - `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth` -> missing
    - `outputs/ai_id_ref/checkpoints/env_research_al31_4_06kw/best_actor.pth` -> missing
    - `outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth` -> missing
  - `config/checkpoint_registry.json` still points to those paths, so the registry is structurally present but currently unresolved
  - only partial AO2 logs/summaries survived from the latest branch (`pilot8b/9/10`); no live `actor_ep*.pth` survived

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
- Latest post-restore closure state (`2026-03-20`):
  - `AIR56` mean Step27 gate is now closed in the canonical live run:
    - `outputs/step27_postrestore_live_round9_ep008_eta60pin_20260320/step27_air56_acceptance.json`
    - `mean_pass=true`
    - live pair used there:
      - checkpoint: `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth` copied from `outputs/air56_round1_subset_ep006_012_20260320/actor_ep008.pth`
      - config envelope: `ai_eval_*` in `config/env_research_air56_025kw.py` aligned to candidate `eta_mid_60_pin`
  - A false-red Step28 reproduce run was identified and explained:
    - `outputs/reproduce_ieee_step28_20260320a/...`
    - root cause: `tools/reproduce_ieee_step28.py` defaulted to `--mic-mode rule` when the run was launched without explicit `--mic-mode ai`
    - that candidate is not scientifically relevant for MIC-AI verification
  - The correct AI reproduce candidate still remains red under strict verify:
    - `paper/ieee_2026/data/step28/20260320_postrestore_ai_ep008_eta60pin/FINAL_CHECKLIST_AUTO.md`
    - `paper/ieee_2026/data/step28/20260320_postrestore_ai_ep008_eta60pin/IEEE_SUBMISSION_DOSSIER.json`
    - real remaining blockers are now narrow:
      - `AIR56`: `mean_pass=true`, but `worst_case_pass=false`
      - `AL31`: `avg_eta_gain_pct_mean=-0.00288`, so `mean_pass=false`
      - `AO2`: `acceptance_pass=true`
    - conclusion:
      - the project is no longer blocked by global AI collapse or packaging errors
      - the remaining work is a strict-verify guardrail closure on `AIR56 worst-case` and `AL31 eta`, not a broad algorithmic rebuild
- Strict-selection alignment update (`2026-03-22`):
  - `mic_ai/ai/train_ai_id_ref.py` now passes the full external Step27 threshold set into `scan_step27_checkpoints.py`, including:
    - aggregate thresholds (`min_avg_power_saving_pct`, `min_avg_eta_gain_pct`, `max_err_failures`, `start_stop`, current ratios)
    - worst-seed thresholds (`*_min_seed`, `err_failures_max_seed`)
  - regression coverage was extended in:
    - `tests/test_train_ai_id_ref_external_step27.py`
  - implication:
    - training-time checkpoint promotion is now aligned with the same strict objective used by the low-compute scan/tune tools
- AIR56 strict-closure update (`2026-03-22`):
  - the current live AIR56 checkpoint remains:
    - `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth`
  - cheap no-training recheck under the exact `Step28` strict metric with `seed_perturbation=0.2` found a much better live envelope than the old `p_in` baseline:
    - `outputs/air56_live_ckpt_specific_power_p02_search_20260322/eta_mid_60_sp/air56_checkpoint_scan_summary.json`
    - metrics:
      - `avg_power_saving_pct = +0.5505%`
      - `avg_eta_gain_pct = -0.00635%`
      - `avg_power_saving_pct_min_seed = +0.49256%`
      - `avg_eta_gain_pct_min_seed = -0.02904%`
    - conclusion:
      - cheap AIR56 closure is now a tiny `eta` tail problem under `p0.2`, not a large tracking/power failure
  - a fresh full candidate was reproduced from the current config:
    - `paper/ieee_2026/data/step28/20260322_postrestore_ai_air56sp/derived_ieee/motor_tuning_acceptance_summary.json`
    - result:
      - `AIR56` still red
      - `AL31` still red
      - `AO2` green
  - AIR56 low-compute checkpoint work was pushed further before declaring the short path exhausted:
    - short warm-start from the live checkpoint with strict external selection:
      - `outputs/air56_warmstart_step28_bridge_20260322/.../external_step27_scan/air56_checkpoint_scan_summary.json`
    - best post-train checkpoint/candidate pair found so far:
      - checkpoint: `outputs/air56_warmstart_step28_bridge_20260322/results_run/20260322_065126_tmp_air56_train_step28_bridge_20260322_ai_id_ref/eval/actor_ep002.pth`
      - candidate: `pin60_hard_b`
      - artifact:
        - `outputs/air56_postft_recheck_20260322/actor_ep002/pin60_hard_b/air56_checkpoint_scan_summary.json`
      - metrics:
        - `avg_power_saving_pct = +0.5964%`
        - `avg_eta_gain_pct = -0.00332%`
        - `avg_power_saving_pct_min_seed = +0.4893%`
        - `avg_eta_gain_pct_min_seed = -0.0212%`
    - a second targeted refinement from that lineage improved the best selector score again but still did not cross zero on `eta`:
      - `outputs/air56_refine_ep002_pin60hb_20260322/.../external_step27_scan/air56_checkpoint_scan_summary.json`
      - best checkpoint there: `actor_ep010.pth`
      - best metrics under the selected candidate:
        - `avg_power_saving_pct = +0.6968%`
        - `avg_eta_gain_pct = -0.00880%`
        - `avg_power_saving_pct_min_seed = +0.6327%`
        - `avg_eta_gain_pct_min_seed = -0.0275%`
    - additional post-train rechecks across nearby `pin/sp` envelopes and one eta-biased micro-refinement did not produce a strict pass
    - conclusion:
      - the AIR56 cheap/short path is now materially exhausted
      - the next justified AIR56 step is a longer eta-aware checkpoint finetune, not more supervisor-only candidate sweeps
- AL31 next-step preparation (`2026-03-22`):
  - no new AL31 checkpoint was promoted yet
  - best cheap candidate remains:
    - `outputs/al31_eta_closure_round2_20260322/al31_tuning_summary.json`
    - tag: `eta_clip_03a`
  - next low-compute checkpoint-level recipe was prepared from:
    - `outputs/al31_rebuild_round1_20260318a/results_run/20260318_111437_env_research_al31_4_06kw_ai_id_ref/eval/actor_ep005.pth`
  - that recipe should be executed only after AIR56 is closed or if AIR56 compute is paused
- AIR56 continuation (`2026-03-22`, later cycle):
  - a longer envelope-aware AIR56 warm-start was executed from the best known `actor_ep002 + pin60_hard_b` lineage:
    - `outputs/air56_envelope_long_20260322/results_run/20260322_075249_tmp_air56_refine_ep002_pin60hb_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - external selection was switched to:
      - `use_envelope_acceptance=true`
      - full strict `p0.2` aggregate + min-seed thresholds
  - result:
    - best checkpoint became `actor_ep009.pth`
    - this is the first AIR56 checkpoint in this branch that reduced the canonical failure count to one remaining row-level failure:
      - `envelope_fail_count = 1`
      - remaining failure is in `speed_step` (`pass_count = 4/5`)
    - but strict aggregate closure is still red:
      - `avg_power_saving_pct = +0.4542%`
      - `avg_eta_gain_pct = -0.00546%`
      - `avg_power_saving_pct_min_seed = +0.3105%`
      - `avg_eta_gain_pct_min_seed = -0.02616%`
  - implication:
    - the longer AIR56 run improved canonical envelope behaviour materially
    - but it did not close the aggregate `power/eta` gate, so the selector now exposes a cleaner tradeoff rather than a solved branch
  - post-train candidate recheck on `actor_ep009`:
    - `outputs/air56_ep009_candidate_recheck_20260322/air56_tuning_summary.json`
    - no candidate achieved `acceptance_pass=true`
    - `sp60_base` / `pin60_base` moved `eta` positive, but both dropped `avg_power_saving_pct` below `0.5`
    - best canonical-envelope candidate remained `pin60_hard_b`
  - post-train candidate recheck on the higher-power checkpoint `actor_ep019`:
    - `outputs/air56_ep019_candidate_recheck_20260322/air56_tuning_summary.json`
    - no candidate achieved `acceptance_pass=true`
    - `sp60_soft_b` fixed `speed_step` locally and reduced the remaining issue to `load_step`, but aggregate metrics still fell below the strict `power/eta` min-seed gate
    - the original `pin60_hard_b` candidate preserved higher `power`, but still failed on `speed_step`
  - conclusion:
    - the cheap AIR56 closure path is now exhausted for the current lineage:
      - longer warm-start with canonical-envelope checkpoint selection did not pass
      - post-train candidate rechecks on the two most promising checkpoints did not pass
    - the next justified AIR56 step is no longer another `pin/sp` local sweep:
      - it must be a materially different training recipe aimed at the last `speed_step` error tail while keeping `power/eta` above the strict min-seed gate
- AL31 execution result (`2026-03-22`, later cycle):
  - the prepared low-compute warm-start recipe was executed:
    - `outputs/al31_warmstart_eta03a_step28_20260322/results_run/20260322_084136_env_research_al31_4_06kw_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
  - outcome:
    - the run did not close AL31
    - the selector promoted `actor_ep000.pth` only because it was the sole checkpoint with `envelope_all_rows_pass=true`
    - but that selected checkpoint still failed strict aggregate acceptance:
      - `avg_eta_gain_pct = -0.00469%`
      - `avg_power_saving_pct_min_seed = -0.2000%`
      - `start_stop_power_saving_pct_min_seed = -0.6455%`
  - more aggressive later checkpoints in the same run improved mean `power/eta`:
    - e.g. `actor_ep019.pth` reached:
      - `avg_power_saving_pct = +1.4026%`
      - `avg_eta_gain_pct = +0.00285%`
    - but it broke canonical `start_stop` robustness (`pass_count = 4/5`) and therefore still failed strict selection
  - conclusion:
    - this first AL31 warm-start recipe is rejected as a closure path
    - current live `actor_ep005` remains a better base than the new run for practical strict closure
    - the next AL31 step, if reopened, must preserve `start_stop` worst-seed robustness rather than using the same random warm-start recipe
- Research infrastructure update (`2026-03-22`, late cycle):
  - `tools/tune_motor_step27.py` now supports explicit offline checkpoint evaluation via:
    - `--checkpoint-path`
  - focused regression coverage was extended in:
    - `tests/test_tune_motor_step27_candidates.py`
  - `mic_ai/ai/train_ai_id_ref.py` now exposes micro-finetune PPO knobs:
    - `--lr`
    - `--entropy-coef`
  - `mic_ai/ai/train_ai_id_ref.py` external Step27 promotion can now optionally rank the warm-start baseline together with new snapshots:
    - `--external-step27-include-init-checkpoint`
  - focused regression coverage was extended in:
    - `tests/test_train_ai_id_ref_external_step27.py`
  - implication:
    - post-train candidate rechecks no longer require registry mutation
    - near-boundary checkpoint nudges can now be attempted without the old fixed optimizer aggressiveness
    - micro-finetune runs no longer have to choose between only the new snapshots if the init checkpoint is still objectively best
- AIR56 late-cycle closure state (`2026-03-22`, latest):
  - explicit-checkpoint strict local search was executed on the best envelope-green checkpoint:
    - checkpoint:
      - `outputs/air56_ep022_mix04_train_20260322/results_run/20260322_102032_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep002.pth`
    - candidate sweep:
      - `outputs/air56_ep002_strict_local_20260322/air56_tuning_summary.json`
    - result:
      - best candidate remained `pin60_mix_04`
      - `envelope_all_rows_pass=true`
      - remaining gap stayed in strict worst-seed aggregate:
        - `avg_power_saving_pct_min_seed = +0.4764%`
        - `avg_eta_gain_pct_min_seed = -0.0229%`
  - a low-lr / low-entropy micro-finetune from that checkpoint was executed:
    - `outputs/air56_ep002_mix04_micro_lr_20260322/results_run/20260322_115034_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - best checkpoint there became `actor_ep001.pth`
    - result:
      - `envelope_all_rows_pass=true`
      - `avg_eta_gain_pct_mean = +0.00143%`
      - but `avg_power_saving_pct_min_seed` collapsed to `+0.3564%`
  - explicit-checkpoint strict local search was then executed on that conservative checkpoint:
    - `outputs/air56_ep001_micro_strict_local_20260322/air56_tuning_summary.json`
    - result:
      - no candidate beat `pin60_mix_04`
      - candidate-only closure remained impossible from that basin
  - a power-biased micro-finetune from `actor_ep001` was also executed:
    - `outputs/air56_ep001_powerbump_micro_20260322/results_run/20260322_121739_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - result:
      - no checkpoint closed the branch
      - the best selected checkpoint lost envelope and still missed worst-seed power/eta
  - after adding `--external-step27-include-init-checkpoint`, a new AIR56 micro-run was executed with the warm-start baseline included in snapshot ranking:
    - `outputs/air56_ep002_includeinit_micro_20260322/results_run/20260322_122733_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - important outcome:
      - the selector still chose a new checkpoint over the init baseline, so the improvement is real and not an artifact of missing baseline comparison
      - new best checkpoint:
        - `actor_ep003.pth`
      - metrics:
        - `avg_power_saving_pct = +0.6340%`
        - `avg_eta_gain_pct = +0.00136%`
        - `avg_power_saving_pct_min_seed = +0.5196%`
        - `avg_eta_gain_pct_min_seed = -0.0195%`
        - `envelope_all_rows_pass = true`
        - `err_failures = 0.0`
    - implication:
      - AIR56 now has a checkpoint that simultaneously keeps canonical envelope green, mean eta positive, and worst-case power above threshold
      - the only remaining AIR56 blocker on this branch is the worst-case eta tail
  - a strict post-train candidate recheck on that new checkpoint was executed:
    - `outputs/air56_ep003_includeinit_strict_local_20260322/air56_tuning_summary.json`
    - result:
      - no candidate beat `pin60_mix_04`
      - the deploy-side search could not remove the remaining worst-case eta tail
  - a final eta-biased micro-finetune was executed from that new best checkpoint with the init checkpoint again included in ranking:
    - `outputs/air56_ep003_etaedge_micro_20260322/results_run/20260322_125714_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - result:
      - the selector kept `actor_ep_init.pth` (the previous best `actor_ep003`) as the best checkpoint
      - no further policy improvement was found on this cheap eta-edge path
  - a higher-power but envelope-red checkpoint from the low-lr run was also rechecked:
    - checkpoint:
      - `outputs/air56_ep002_mix04_etafinetune_20260322/results_run/20260322_110416_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep003.pth`
    - candidate sweep:
      - `outputs/air56_ep003_strict_local_20260322/air56_tuning_summary.json`
    - result:
      - no candidate restored full envelope while preserving the improved power headroom
  - conclusion:
    - AIR56 is now scientifically narrowed to a policy-level tradeoff, not a deploy-parameter problem
    - current best known AIR56 pair is now:
      - checkpoint: `outputs/air56_ep002_includeinit_micro_20260322/results_run/20260322_122733_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep003.pth`
      - candidate: `pin60_mix_04`
    - remaining AIR56 gap is now strictly:
      - `avg_eta_gain_pct_min_seed = -0.0195%`
    - next justified AIR56 step is no longer another candidate sweep:
      - it must be a truly selective low-lr policy refinement path, ideally with the init checkpoint included in the external snapshot ranking so micro-runs cannot regress by construction
- AL31 late-cycle closure state (`2026-03-22`, latest):
  - explicit-checkpoint bridge search was executed on the best surviving checkpoint:
    - checkpoint:
      - `outputs/al31_rebuild_round1_20260318a/results_run/20260318_111437_env_research_al31_4_06kw_ai_id_ref/eval/actor_ep005.pth`
    - bridge sweep:
      - `outputs/al31_ep005_bridge_search_20260322/al31_tuning_summary.json`
    - result:
      - `al31_bridge_05` moved `avg_eta_gain_pct_mean` above zero, but reopened worst-case / envelope tails too aggressively
  - a tighter midpoint search was then executed:
    - `outputs/al31_ep005_bridge_mid_20260322/al31_tuning_summary.json`
    - best candidate:
      - `al31_mid_04`
    - metrics:
      - `envelope_all_rows_pass=true`
      - `avg_eta_gain_pct_mean = +0.000554%`
      - `avg_eta_gain_pct_min = -0.004466%`
      - `start_stop_power_saving_pct_min = +0.8148%`
    - implication:
      - the original AL31 mean-pass blocker is closed on this candidate
      - the remaining AL31 blocker is now only the worst-case eta tail
  - a low-lr micro-finetune around `al31_mid_04` was executed:
    - `outputs/al31_mid04_micro_lr_20260322/results_run/20260322_122222_tmp_al31_mid04_train_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - result:
      - no checkpoint closed the remaining `avg_eta_gain_pct_min >= 0` guardrail
      - best envelope-green snapshot (`actor_ep004.pth`) improved power headroom but worsened eta to `avg_eta_gain_pct_mean = -0.00343%`
  - conclusion:
    - AL31 is now much closer than before:
      - mean eta is already positive on `al31_mid_04`
      - start/stop robustness remains green
    - the remaining AL31 gap is tiny and localized:
      - `avg_eta_gain_pct_min = -0.004466%`
    - next justified AL31 step is not another broad warm-start:
      - either a tiny deploy-side eta micro-search around `al31_mid_04`
      - or a final canonical Step28 reproduce check once AIR56 is also ready, to confirm that no additional pipeline-level averaging already resolves this tail
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
- AO2 checkpoint-level update (`2026-03-17`):
  - A new short warm-start pilot from the current AO2 checkpoint materially improved the external Step27 profile, but still did not pass the full gate:
    - `outputs/ao2_ft_pilot1_20260317/checkpoint_eval/new_best/ao2_tuning_summary.json`
    - best result moved to `avg_power_saving_pct=4.116%`, `avg_eta_gain_pct=9.513%`, `err_failures=2.333`, `start_stop_power_saving_pct=13.822%`
    - this is much closer than the old AO2 baseline (`8.520% / 17.445% / 2.667 / 31.989%`), but it still fails on `err_failures` and current peaks
  - A targeted guardrail sweep around that new AO2 checkpoint did not beat the same `base_current` candidate:
    - `outputs/ao2_ft_pilot1_20260317/checkpoint_eval/new_best_guardrail_sweep/ao2_tuning_summary.json`
  - A dedicated external snapshot-scan tool was added so actor snapshots can be ranked against Step27 directly instead of relying on the trainer's internal score:
    - `tools/scan_step27_checkpoints.py`
    - `outputs/ao2_ft_pilot1_20260317/checkpoint_scan_tool_seed101/ao2_checkpoint_scan_summary.json`
  - Cheap `seed101` snapshot ranking turned out to be unreliable for AO2:
    - the one-seed scan preferred early aggressive snapshots such as `actor_ep007.pth`, but full `3-seed` evaluation still favored the later `new_best` checkpoint
    - conclusion: AO2 checkpoint selection must use the external Step27 gate on the real seed set, not a single-seed proxy
  - Full `3-seed` external scan of all `pilot1` snapshots confirmed that there is no hidden passing AO2 checkpoint inside the short warm-start run:
    - `outputs/ao2_ft_pilot1_20260317/checkpoint_scan_tool_3seed/ao2_checkpoint_scan_summary.json`
    - best external snapshot was `actor_ep019.pth`, matching the already known `new_best` profile (`4.116% / 9.513% / 2.333 / 13.822%`)
  - A second-stage AO2 finetune with small current penalty (`w_current=0.2`) did not improve the selected checkpoint:
    - `outputs/ao2_ft_pilot2_currentpen_20260317/checkpoint_eval/best/ao2_tuning_summary.json`
    - the resulting `best_actor.pth` was byte-identical to the prior AO2 `best_actor.pth`
  - Full `3-seed` external scan of all `pilot2_currentpen` snapshots also confirmed no improvement:
    - `outputs/ao2_ft_pilot2_currentpen_20260317/checkpoint_scan_tool_3seed/ao2_checkpoint_scan_summary.json`
    - best external snapshot remained `actor_ep000.pth`, i.e. the incoming checkpoint before the current-penalty finetune
  - A robustness-oriented AO2 finetune from the best external snapshot using randomized `omega/load` training conditions also failed to beat the incoming checkpoint:
    - `outputs/ao2_ft_pilot3_randrobust_20260317/checkpoint_scan_tool_3seed/ao2_checkpoint_scan_summary.json`
    - best external snapshot again remained `actor_ep000.pth`, while later snapshots drifted below zero on energy/start-stop metrics
  - A longer mild-randomized AO2 finetune from the best external snapshot also failed to improve the external acceptance profile:
    - `outputs/ao2_ft_pilot4_mildrand_20260317/checkpoint_scan_tool_3seed/ao2_checkpoint_scan_summary.json`
    - best external snapshot again remained `actor_ep000.pth` (`4.116% / 9.513% / 2.333 / 13.822%`)
  - Conclusion: the cheap and medium-budget AO2 path is now exhausted; the next real step is a materially different longer AO2 retrain cycle with external checkpoint selection against the full Step27 acceptance objective.
- AO2 research continuation (`2026-03-18`):
  - The live workspace under `C:\mic_theory` lost the code tree and retained only `outputs/`; a stable git root was restored separately at `C:\mt` so research can continue in a controlled environment.
  - The restored git root was behind the latest local research state; the missing trainer/selector patches have now been re-applied in `C:\mt`.
  - Joint `checkpoint x candidate` search was closed for the surviving `pilot1` AO2 snapshots:
    - the full `3-seed` pairwise re-check still favored `actor_ep019.pth + base_current`
    - no alternative candidate/ checkpoint pair beat the known `4.116% / 9.513% / 2.333 / 13.822%` profile under the real `p0.2` objective
  - A current-penalized AO2 warm-start with external Step27 checkpoint selection materially improved the best live result:
    - `outputs/ao2_ft_pilot8b_ep019_wcurrent05_extsel_20260317/.../external_step27_selection.json`
    - selected `actor_ep013.pth`
    - best metrics moved to `avg_power_saving_pct=5.444%`, `avg_eta_gain_pct=5.566%`, `err_failures=2.333`, `start_stop_power_saving_pct=19.751%`, `worst_current_peak_ratio=1.379`
    - this improved `power`, `start_stop`, and `peak` versus the old `actor_ep019 + base_current` baseline, but still failed the gate on `err_failures` and current peaks
  - A full candidate-grid re-check around that improved `actor_ep013.pth` snapshot did not find a better supervisor/id_ref setting:
    - `outputs/ao2_pilot8b_ep013_candidate_grid_3seed_20260318/ao2_checkpoint_candidate_grid_summary.json`
    - `base_current` remained the best candidate for `actor_ep013.pth`
  - Two follow-up AO2 runs uncovered an engineering failure mode in the external selector:
    - `outputs/ao2_ft_pilot9_ep009_peakbias_extsel_20260318/train.log`
    - `outputs/ao2_ft_pilot10_ep007_supalign_tracktight_20260318/run.log`
    - both runs crashed during external checkpoint selection when a snapshot path such as `actor_ep006.pth` disappeared between collection and loading
    - `tools/scan_step27_checkpoints.py` is now hardened to skip missing snapshots instead of crashing
  - The partially completed `pilot10` log still showed the best AO2 profile seen so far in this branch before the selector crash:
    - `avg_power_saving_pct=4.441%`, `avg_eta_gain_pct=4.490%`, `err_failures=2.333`, `start_stop_power_saving_pct=15.155%`, `worst_current_peak_ratio=1.365`
    - this is better than the previous supervisor-aware control point (`4.110% / 4.345% / 2.333 / 14.844% / 1.396`) on `power`, `start_stop`, and `peak`, but the final selection artifact was not produced because the scan crashed
  - A second infrastructure issue appeared after the workspace loss:
    - the latest AO2 checkpoint files for `pilot8/9/10` are no longer present in the surviving `outputs/` tree
    - only logs and summaries remain for those runs
    - the next AO2 rerun must therefore rebuild a live checkpoint lineage from a reproducible starting point inside the restored repo/workspace
  - Trainer capability restoration completed on `2026-03-18`:
    - `mic_ai/ai/train_ai_id_ref.py` now contains the restored scenario-range handling, training-time supervisor hook, and external Step27 checkpoint promotion path
    - new regression: `tests/test_train_ai_id_ref_external_step27.py`
    - external selection now fails explicitly if no evaluated checkpoint exists, instead of accidentally resolving an empty path
  - AO2 live-lineage rebuild completed in `C:\mt`:
    - cold-start `AO2` pilots were run because no surviving AO2 `.pth` remained in the restored workspace
    - the best current live AO2 pair found is:
      - checkpoint: `outputs/ao2_rebuild_pilot3_stableft_20260318/results_run/.../eval/actor_ep005.pth`
      - envelope candidate: `jitter_004`
      - source artifact:
        - `outputs/ao2_rebuild_pilot3_manualsafe02_localsearch_20260318/ao2_manualsafe02_localsearch_summary.json`
      - aggregate `3-seed + p0.2` metrics:
        - `avg_power_saving_pct = +0.0086%`
        - `avg_eta_gain_pct = +0.1983%`
        - `err_failures = 1.0`
        - `start_stop_power_saving_pct = +0.0690%`
        - `worst_current_peak_ratio = 1.1855`
    - the AO2 checkpoint was copied back to the standard registry path:
      - `outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth`
    - `config/env_research_ao2_32_4_3kw.py` was updated to the restored live AO2 envelope so the standard Step27 pipeline can reproduce the same mean metrics
  - AO2 canonical verification status after that rebuild:
    - `python tools/step27_pipeline.py --motors ao2 ... --seed-perturbation --seed-perturb-level 0.2`
      - `outputs/step27_ao2_rebuild_verify_20260318/step27_final_pi_vs_foc_vs_mic.csv`
      - reproduced the aggregate AO2 metrics from the local scan
    - but canonical envelope closure is still red:
      - `outputs/step27_ao2_rebuild_verify_20260318/acceptance_envelopes/acceptance_envelope_summary.json`
      - `all_rows_pass = false`
      - failing scenarios: `speed_step`, `ramp`, `load_step`
    - scenario-level failure pattern:
      - `speed_step`: `power_saving_pct_min < -0.5`
      - `load_step`: `power_saving_pct_min < -0.5` and `eta_gain_pct_min < -0.5`
      - `ramp`: `eta_gain_pct_min < -0.5` and some rows lose `err_ok`
    - conclusion:
      - aggregate Step27 pass is not sufficient; the next AO2 search must optimize the per-scenario envelope directly
  - Envelope-aligned selector closure completed on `2026-03-18`:
    - `tools/tune_motor_step27.py` and `tools/scan_step27_checkpoints.py` now compute and rank by canonical per-scenario envelope, not only aggregate means
    - regression coverage now protects the ranking layer directly:
      - `tests/test_scan_step27_checkpoints.py`
      - `tests/test_tune_motor_step27_candidates.py`
    - canonical re-scan of the current live AO2 pair confirmed that the old aggregate-green result was false-green:
      - `outputs/ao2_rebuild_pilot3_envelope_selector_scan_20260318/ao2_checkpoint_scan_summary.json`
      - current live pair `actor_ep005 + current_cfg(jitter_004)` remains red with:
        - `envelope_fail_count = 9`
        - `envelope_scenario_fail_count = 3`
        - `envelope_gap_total = 4.5836`
      - failing scenarios remain:
        - `speed_step`
        - `ramp`
        - `load_step`
  - AO2 cheap supervisor-only closure was re-tested under the aligned selector on `2026-03-18`:
    - bounded DOE round 1:
      - `outputs/ao2_envelope_targeted_tune_20260318a/ao2_tuning_summary.json`
    - bounded DOE round 2:
      - `outputs/ao2_envelope_targeted_tune_round2_20260318a/ao2_tuning_summary.json`
    - both rounds converged to the same best cheap candidate on the current live checkpoint:
      - tag: `eff_02`
      - metrics:
        - `envelope_fail_count = 8`
        - `envelope_scenario_fail_count = 3`
        - `envelope_gap_total = 4.1800`
        - `avg_power_saving_pct = -0.1404%`
        - `avg_eta_gain_pct = -0.3233%`
        - `err_failures = 1.0`
      - scenario pattern for `eff_02`:
        - `speed_step` improved from `0/3` to `1/3` pass
        - `load_step` still fails `3/3`
        - `ramp` still fails `3/3`, now mainly due to `eta`
      - conclusion:
        - cheap supervisor/id_ref tuning on the current live AO2 checkpoint is now scientifically exhausted
        - the next AO2 step must be checkpoint-level finetuning or retraining with canonical-envelope external selection
  - AO2 canonical closure was then recovered on `2026-03-18` with a compute-capped warm-start retrain:
    - the AO2 config was updated to the best live pre-retrain envelope candidate:
      - `config/env_research_ao2_32_4_3kw.py`
      - best pre-retrain candidate tag: `ug2_relax_1`
      - last cheap-tuning artifact before retrain:
        - `outputs/ao2_targeted_envelope_search12_lastmile_p02_20260318/ao2_tuning_summary.json`
      - this reduced AO2 to one remaining failing `ramp` row under canonical `p0.2`
    - a warm-start AO2 retrain was run from the rebuilt `actor_ep005.pth` lineage using:
      - `mic_ai.ai.train_ai_id_ref`
      - `--fast`
      - `--init-checkpoint outputs/ao2_rebuild_pilot3_stableft_20260318/results_run/.../actor_ep005.pth`
      - external canonical-envelope checkpoint selection on `seed_perturbation=0.2`
      - output root:
        - `outputs/ao2_retrain_ug2relax1_warmstart_20260318/`
    - external selection chose:
      - checkpoint: `actor_ep028.pth`
      - artifact:
        - `outputs/ao2_retrain_ug2relax1_warmstart_20260318/results_run/20260318_094614_env_research_ao2_32_4_3kw_ai_id_ref/external_step27_selection.json`
      - selected checkpoint metrics:
        - `avg_power_saving_pct = +0.1395%`
        - `avg_eta_gain_pct = +0.0508%`
        - `err_failures = 0.0`
        - `start_stop_power_saving_pct = +0.2438%`
        - `worst_current_peak_ratio = 1.2138`
        - `envelope_all_rows_pass = true`
    - the selected AO2 checkpoint was promoted back into the standard registry path:
      - `outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth`
    - standard AO2 Step27 verification now reproduces the green result:
      - `outputs/step27_ao2_post_warmstart_verify_20260318/step27_final_pi_vs_foc_vs_mic.csv`
      - `outputs/step27_ao2_post_warmstart_verify_20260318/acceptance_envelopes/acceptance_envelope_summary.json`
      - `all_rows_pass = true`
      - all four AO2 scenarios now pass `3/3`
  - AO2 canonical closure was re-established again on `2026-03-18` inside the restored `C:\mt` branch after a deeper alignment pass:
    - the remaining AO2 tail was reduced to one row on `actor_ep010 + micro_tight_01`:
      - `envelope_fail_count = 1`
      - `envelope_gap_total = 0.2059`
      - only failing row: `start_stop power_saving_pct_min = -0.7059`
    - cheap residual searches were exhausted and documented:
      - micro local candidate search around `blend_03_softbias`
      - cross-scan of all surviving `round1` AO2 snapshots with the improved candidate set
      - short power-biased and dynamic-biased continuation runs
      - soft-idle supervisor probe with a new `idle_blend` capability
      - none of those cheap paths closed the final row
    - a train/eval misalignment was then identified:
      - current `config/env_research_ao2_32_4_3kw.py` had a dead training supervisor (`update=98`, `dither=0`, `bias_step=0`, `bias_max=0`)
      - the continuation runs were therefore training under a different supervisor regime than the external canonical selector (`micro_tight_01`)
    - after aligning the training supervisor with the best eval candidate, a longer warm-start continuation finally produced a passing checkpoint:
      - run:
        - `outputs/ao2_envelope_ft_round6_alignedsup_long_20260318a/`
      - selected checkpoint:
        - `actor_ep003.pth`
      - selected canonical metrics:
        - `avg_power_saving_pct = +0.1037%`
        - `avg_eta_gain_pct = +0.2483%`
        - `err_failures = 0.0`
        - `start_stop_power_saving_pct = +0.2170%`
        - `envelope_all_rows_pass = true`
        - `acceptance_pass = true`
      - source artifact:
        - `outputs/ao2_envelope_ft_round6_alignedsup_long_20260318a/results_run/20260318_105823_tmp_ao2_train_sup_microtight_20260318_ai_id_ref/external_step27_scan/ao2_checkpoint_scan_summary.json`
    - the selected AO2 checkpoint was promoted into the standard registry path:
      - `outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor.pth`
      - backup:
        - `outputs/ai_id_ref/checkpoints/env_research_ao2_32_4_3kw/best_actor_before_round6_20260318.pth`
    - `config/env_research_ao2_32_4_3kw.py` was updated to the promoted live AO2 pair (`micro_tight_01`)
    - live AO2 Step27 verification via the standard pipeline now confirms row-level green status on the promoted config:
      - run:
        - `outputs/step27_ao2_liveverify_round6_20260318a/`
      - aggregate live metrics (`5` seeds):
        - `avg_power_saving_pct_mean = +0.1684%`
        - `avg_eta_gain_pct_mean = +0.3188%`
        - `err_failures_mean = 0.0`
        - `start_stop_power_saving_pct_mean = +0.6941%`
      - direct canonical row check on the promoted live run:
        - `load_step`: pass
        - `ramp`: pass
        - `speed_step`: pass
        - `start_stop`: pass
        - `all_rows_pass = true`
  - Full `3-motor` post-restore rebaseline was attempted immediately after the AO2 promotion:
    - run:
      - `outputs/step27_postrestore_rebaseline_round6_20260318a/`
    - AO2 is no longer the first blocker in W1
    - AIR56 is still red on the current live config even before reaching AL31:
      - seed-level partial results from the aborted rebaseline:
        - `seed101`: `mic_avg_power=-0.038%`, `mic_start_stop=-1.908%`
        - `seed202`: `mic_avg_power=-0.225%`, `mic_start_stop=-2.266%`
        - `seed303`: `mic_avg_power=+0.001%`, `mic_start_stop=-2.114%`
        - `seed404`: `mic_avg_power=-0.127%`, `mic_start_stop=-1.800%`
        - `seed505`: `mic_avg_power=-0.122%`, `mic_start_stop=-1.521%`
    - the run then stopped on infrastructure, not on AO2:
      - `AL31` has no live checkpoint in the standard registry path
      - `tools/step27_pipeline.py` now fails at load time with:
        - `FileNotFoundError: Checkpoint not found in resolved candidates ... env_research_al31_4_06kw`
    - updated W1 blocker order after AO2 closure:
      - `1.` restore or rebuild live `AL31` checkpoint lineage
      - `2.` close `AIR56` canonical envelope on the current live branch
      - `3.` rerun full `3-motor` Step27, Step28, verify/freeze/promote
  - AL31 lineage was restored on `2026-03-18` via direct cold-start rebuild:
    - run:
      - `outputs/al31_rebuild_round1_20260318a/`
    - external selection chose:
      - `actor_ep005.pth`
    - selected metrics:
      - `envelope_all_rows_pass = true`
      - `avg_power_saving_pct = +1.4993%`
      - `avg_eta_gain_pct = -0.0029%`
      - `err_failures = 0.0`
      - `start_stop_power_saving_pct = +5.7969%`
    - the restored AL31 checkpoint was promoted into:
      - `outputs/ai_id_ref/checkpoints/env_research_al31_4_06kw/best_actor.pth`
    - conclusion:
      - AL31 is no longer missing and is row-level green; only a microscopic aggregate eta tail remains if strict aggregate `eta>=0` is enforced
  - Full `3-motor` post-restore rebaseline after `AO2+AL31` restoration:
    - run:
      - `outputs/step27_postrestore_rebaseline_round6b_20260318a/`
    - canonical row status:
      - `AIR56`: red
      - `AL31`: green
      - `AO2`: green
    - direct envelope check on the run:
      - `all_motors_all_rows_pass = false`
      - the only remaining red motor is `AIR56`
  - AIR56 closure status after the integrated rebaseline:
    - pipeline-internal AIR56 tuning was rerun on the same branch:
      - `outputs/step27_postrestore_with_air56_tune_round6c_20260318a/`
    - selected candidate remained:
      - `manual_air56_step27_fix_01`
    - this candidate fixes the energy/start-stop side but still fails canonical rows on tracking:
      - `load_step`: `err_ok_all = false`
      - `ramp`: `err_ok_all = false`
      - `start_stop`: `err_ok_all = false`
    - a follow-up envelope-aware aggressive/no-idle AIR56 candidate sweep was also run:
      - `outputs/air56_tracking_search_p02_20260318a/`
    - result:
      - every tested AIR56 candidate still had `err_failures = 3.0`
      - cheap supervisor/id_ref search is now exhausted for AIR56 as well
  - AIR56 registry/provenance gap found on `2026-03-19`:
    - the live registry checkpoint
      - `outputs/ai_id_ref/checkpoints/env_research_air56_025kw/best_actor.pth`
      - does **not** match the externally selected AIR56 checkpoint from the rebuild run
      - `outputs/air56_rebuild_basecurrent_20260318/results_run/20260318_102400_env_research_air56_025kw_ai_id_ref/best_actor_step27.pth`
    - the AIR56 rebuild run had already found a materially better checkpoint under the canonical selector:
      - `actor_ep004.pth` / `best_actor_step27.pth`
      - `outputs/air56_rebuild_basecurrent_20260318/results_run/20260318_102400_env_research_air56_025kw_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
      - on `3` seeds with canonical envelope and `base_current`, it reduced AIR56 down to one remaining failing row:
        - `envelope_fail_count = 1`
        - only `speed_step` remained red
    - root cause:
      - `mic_ai/ai/train_ai_id_ref.py` promoted the externally selected checkpoint only into the per-run artifact `best_actor_step27.pth`
      - the standard registry path `outputs/ai_id_ref/checkpoints/.../best_actor.pth` remained on the old trainer-best checkpoint
    - fix:
      - trainer now copies the externally selected Step27 checkpoint into the standard registry best path after selection
      - regression:
        - `tests/test_train_ai_id_ref_external_step27.py`
    - implication:
      - before any new AIR56 retrain, the next cheap step is to re-evaluate `best_actor_step27.pth` on the full `5`-seed envelope and, if needed, run only a narrow candidate bridge search around it
    - updated W1 blocker order:
      - `1.` checkpoint-level AIR56 retrain / policy update focused on tracking closure
      - `2.` rerun full `3-motor` Step27/acceptance with the promoted AIR56 checkpoint
      - `3.` rebuild Step28, verify/freeze/promote the new post-restore tag
- Passport/package update (`2026-03-17`):
  - The post-restore passport gap was traced to the reproduce/package path, not to missing motor configs:
    - `tools/reproduce_ieee_step28.py` only built passport artifacts when `--build-passport` was requested
    - `scripts/package_ieee_step28.py` only copied passport artifacts when `--passport-dir` was explicitly passed
  - The reproduce pipeline was updated so passport artifacts are built by default unless `--no-build-passport` is used.
  - Smoke coverage was updated to require `passport/passport_compare_3motors.{csv,md,json}` in the packaged output.
  - The old `paper/ieee_2026/data/step28/20260308_postrestore_ai` candidate was rebuilt with a real `passport/` block and downstream checklist/candidate/dossier/verify files were refreshed.
  - `verification_ok` still remains red, but passport is no longer one of the missing package elements; the remaining blockers are acceptance/guardrails.

Work:
- [x] Rebuild the missing passport block for the post-restore candidate so the package no longer skips passport checks.
- [x] Re-run low-compute AL31/AO2/AIR56 tuning under the current envelope constraints instead of relying only on the old recovered checkpoint set.
- [x] Add targeted-candidate support to the tuning tool so local refinement can be evaluated without random sweeps.
- [x] Run a short AO2 warm-start pilot from the recovered checkpoint to test whether cheap checkpoint adaptation is sufficient.
- [x] Add a reusable external Step27 checkpoint-scan tool so actor snapshots can be selected by the real acceptance objective.
- [x] Run AO2 low-budget checkpoint selection experiments (`new_best`, `new_last`, selected `actor_epXXX`, guardrail sweep, cheap current-penalty finetune).
- [x] Fix the Step28 reproduce/package pipeline so passport artifacts are built and packaged by default.
- [x] Run explicit checkpoint-level finetuning/retraining for AO2 with external canonical-envelope snapshot selection.
- [x] For AO2, couple finetune/retrain with external snapshot selection on the real Step27 seed set before promoting any checkpoint.
- [x] Restore the missing trainer/selector patches into the stable git root (`C:\mt`) so the recovered workspace matches the latest local research capabilities.
- [x] Harden `tools/scan_step27_checkpoints.py` so external selection skips missing checkpoints instead of crashing mid-scan.
- [x] Re-establish a live AO2 checkpoint lineage in the restored workspace, then rerun the next supervisor-aware/current-aware AO2 experiment from that reproducible starting point.
- [x] Decide the cheapest scientifically valid AO2 rebuild path now that no live checkpoint exists in the restored repo:
  - direct `train_ai_id_ref.py` cold-start rebuild was the only reproducible path in `C:\mt`
  - `train_3motors_pipeline.py` was not used because no reusable manifest/checkpoint lineage survived
- [x] Align `tools/scan_step27_checkpoints.py` / external training selection with canonical per-scenario envelope criteria instead of aggregate-only means.
- [x] Run bounded AO2 supervisor-only searches against the envelope-aligned selector on the rebuilt live checkpoint:
  - `outputs/ao2_envelope_targeted_tune_20260318a/`
  - `outputs/ao2_envelope_targeted_tune_round2_20260318a/`
- [x] Run the next AO2 checkpoint-level finetune/retrain against the envelope-aligned selector and promote only the externally selected canonical-envelope checkpoint.
- [ ] Re-run baseline Step27 for the selected post-restore checkpoints.
- [ ] Re-run acceptance envelopes and identify scenario-level failures by motor:
  - AIR56: `load_step`, `speed_step`
  - AL31: `load_step`, `speed_step`, `start_stop`
- [x] Re-run AO2 baseline Step27 plus canonical acceptance after the promoted warm-start checkpoint.
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
- [ ] Finish restoring the latest local W1 trainer/selector capabilities into the stable git root at `C:\mt`.
- [ ] Fix the external Step27 selector crash-path on missing checkpoints and re-run the interrupted AO2 branch from a reproducible live checkpoint lineage.
- [ ] After the AO2 branch is back to a stable reproducible state, resume post-restore envelope closure and only then return to onboarding/refactor work.

## Update rule
- Only this file may serve as the active root master plan.
- Every update must record:
  - what was closed
  - what remains blocked
  - which artifact proves the claim
- If a track is intentionally retired instead of finished, document that decision explicitly here.
