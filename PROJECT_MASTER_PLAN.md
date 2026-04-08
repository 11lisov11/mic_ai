# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-04-08`
Repository: `C:\mic_theory`
Last pushed commit: `eaf8d78`

## Canonical source
- This file is the only active master plan allowed in the repository root.
- Historical root plans and execution logs must stay only under `docs/plan_archive/`.
- `C:\mic_theory` is the only canonical working repository.
- `C:\mt` and `C:\mic_theory_repo_restored` are not canonical roots.

## Current factual snapshot
- Git state is currently clean on branch `main` after push `eaf8d78`.
- Latest confirmed smoke in the current cycle is green:
  - `pytest -q tests/test_root_hygiene_smoke.py tests/test_report_plan_completion_smoke.py`
  - `3 passed`
- Frozen release is already green and must not be reopened unless a real bug is found:
  - `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/VERIFY_SUBMISSION_CANDIDATE.json`
  - `verification_ok=true`
- Universal onboarding is implemented and correctness-gated, but not energy-closed:
  - correctness green:
    - `outputs/train_any_motor_pipeline/eval_demo_any_reval_gate_default2/any_motor_onboarding_report.json`
    - `all_ok=true`
  - energy still red:
    - `outputs/train_any_motor_pipeline/eval_demo_any_plan_v2/any_motor_onboarding_report.json`
    - bottleneck: `ao2 power_saving_pct_mean < 0`
- Post-restore research branch is still not closed to submission-ready state:
  - `paper/ieee_2026/data/step28/20260320_postrestore_ai_ep008_eta60pin/FINAL_CHECKLIST_AUTO.md`
  - `paper/ieee_2026/data/step28/20260322_postrestore_ai_air56sp/FINAL_CHECKLIST_AUTO.md`
  - strict verify is still red because `AIR56` and `AL31` are not fully closed on worst-case energy tails
- Historical green candidate `20260304_al31_robust_rand009_nodrift_v3` is not direct W1 proof:
  - it was built with `seed_perturbation=false`
  - it is useful only as provenance/recovery context, not as current strict `p0.2` closure evidence
- `AO2` is no longer the blocker in W1.
- Current codebase now includes new reward-alignment and training-basin controls:
  - `mic_ai/ai/ai_env.py`
  - added soft energy reward gate for `ai_id_ref` / `ai_current`
  - added running-eta penalty / episode-energy knobs
  - this is code-complete and tested
  - runtime validation is now partially complete: reward-only micro-runs were validated and found insufficient
  - `mic_ai/ai/train_ai_id_ref.py`
  - added energy curriculum, CLI reward overrides, hidden-size override, hidden-size inference from warm-start checkpoints, mixed-width checkpoint adaptation, and external Step27 resume plumbing
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
1. W1 post-restore strict closure is not finished.
2. A new green post-restore Step28 candidate has not been produced.
3. Universal any-motor energy gate is not closed.
4. Universal onboarding is not yet proven on a real identification-first flow.
5. Main orchestration scripts are still too monolithic.
6. Repository hygiene is not finished:
   - `README.md` still needs UTF-8 normalization and content cleanup.
   - archive placement / root hygiene rule still needs to be documented outside the test itself.
7. Final test coverage is still incomplete for the newest onboarding and reward-alignment modes.

## Definition of 100% done
The project is finished only when all items below are true.

- A new post-restore candidate exists with:
  - `ready_for_submission=true`
  - `checklist_ready_for_submission=true`
  - `verification_ok=true`
- W1 strict closure is green for all three motors in the live post-restore branch:
  - `AIR56` green on mean and worst-case criteria
  - `AL31` green on mean and worst-case criteria
  - `AO2` green
- Universal onboarding has:
  - benchmark correctness green
  - energy gate green
  - documented passport-only flow
  - documented passport + identification flow
  - reproducible artifacts for both
- Active orchestration scripts are refactored to smaller responsibilities:
  - `tools/step27_pipeline.py`
  - `tools/train_any_motor_pipeline.py`
  - `tools/train_3motors_pipeline.py`
  - `tools/robust_motor_hardening.py`
- Documentation is clean and current:
  - `README.md` readable in UTF-8
  - runbooks match actual entrypoints and gates
  - only one active root plan remains
- `pytest -q` is green after all cleanup.

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

#### W1.1 Current motor status

`AO2`
- status: green in the live research branch
- blocker status: closed
- do not spend more compute on `AO2` unless a regression appears in the final full run

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
  - newest active branch is now the strict power-recovery continuation from `actor_ep019`:
    - `outputs/air56_ep019_powerrecover_20260408b/results_run/20260408_153717_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_progress.json`
    - live signal so far:
      - checkpoints `actor_ep002` through `actor_ep007` already pass strict aggregate gates
      - failure mode is no longer aggregate power / eta; it is row-level `load_step` tracking (`err`)
      - strongest balance so far is `actor_ep006`:
        - `avg_power_saving_pct = 0.8630392380153374`
        - `avg_eta_gain_pct = 0.10702947229523918`
        - `avg_power_saving_pct_min_seed = 0.7322733162081374`
        - `avg_eta_gain_pct_min_seed = 0.08032673473104546`
        - `load_step pass_count = 2/5`
      - highest aggregate margin so far is `actor_ep007`, but with worse `load_step` robustness:
        - `load_step pass_count = 1/5`
  - current execution rule:
    - do not start another retrain until the active strict scan above finishes
    - if no envelope-clean winner appears, run one targeted local-safe retune around the best aggregate-passing checkpoint from this branch, prioritizing:
      - `load_step pass_count`
      - then `avg_power_saving_pct_min_seed`
      - then `avg_eta_gain_pct_min_seed`
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
  - the incumbent and reltrack frontiers therefore remain complementary:
    - incumbent is envelope-clean but eta-tail red
    - reltrack frontier nearly closes eta-tail but still drops one `speed_step` row

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
- `AIR56` strict pass:
  - `avg_power_saving_pct > 0.5`
  - `avg_eta_gain_pct >= 0`
  - `avg_power_saving_pct_min_seed > 0.5`
  - `avg_eta_gain_pct_min_seed >= 0`
  - `envelope_all_rows_pass = true`
- `AL31` strict pass:
  - `avg_power_saving_pct >= 0`
  - `avg_eta_gain_pct >= 0`
  - `avg_power_saving_pct_min_seed >= 0`
  - `avg_eta_gain_pct_min_seed >= 0`
  - `envelope_all_rows_pass = true`
- `AO2` remains green in the same final full run

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

## Current next command order
This is the strict next execution sequence from the current state.

1. Freeze the current best local pairs as the factual baselines:
   - `AL31`: `actor_ep_init + mid04_speed_dn_04`
   - `AIR56` strict incumbent: `actor_ep005 + mix04_base`
   - `AIR56` near-pass frontier: `actor_ep_init + etaep_bias_dn_2`
2. Do not spend more compute on exhausted candidate-only retunes for those baselines.
3. Keep `AL31` parked as a tiny-tail blocker; do not spend more compute on near-identical `ai_id_ref` continuations until `AIR56` changes the global context.
4. For `AIR56`, do not reopen exhausted reserve/control-family probes:
   - `ai_current`
   - `ai_voltage`
   - `ai_speed`
   - `foc_assist`
5. Next `AIR56` attempt must change the algorithmic basin or optimization regime itself while preserving the known reltrack/incumbent lessons.
   - first preferred branch:
     - medium-budget `ai_id_ref` basin/objective shift from the strict incumbent lineage
     - external selection only against the strict incumbent deployment candidate `mix04_base`
     - strict W1 thresholds:
       - `avg_power_saving_pct >= 0.5`
       - `avg_eta_gain_pct >= 0.0`
       - `avg_power_saving_pct_min_seed >= 0.5`
       - `avg_eta_gain_pct_min_seed >= 0.0`
       - `envelope_all_rows_pass = true`
6. Only after both motors are green, rerun full Step27 and Step28.
7. After W1 closes, move to onboarding-energy closure, then refactor/docs/tests.

