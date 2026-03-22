# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-03-22`
Repository: `C:\mic_theory`
Last pushed commit: `a22d4df`

## Canonical source
- This file is the only active master plan allowed in the repository root.
- Historical root plans and execution logs must stay only under `docs/plan_archive/`.
- `C:\mic_theory` is the only canonical working repository.
- `C:\mt` and `C:\mic_theory_repo_restored` are not canonical roots.

## Current factual snapshot
- Git state is clean after push:
  - branch: `main`
  - working tree: clean
- Latest focused regression after reward-gate change is green:
  - `pytest -q tests/test_ai_env_reward_gate.py tests/test_train_ai_id_ref_external_step27.py tests/test_tune_motor_step27_candidates.py tests/test_scan_step27_checkpoints.py`
  - `17 passed`
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
- `AO2` is no longer the blocker in W1.
- Current codebase now includes a new reward-alignment capability:
  - `mic_ai/ai/ai_env.py`
  - added soft energy reward gate for `ai_id_ref` / `ai_current`
  - this is code-complete and tested, but runtime research validation is still pending

## What is still not finished to 100%
1. W1 post-restore strict closure is not finished.
2. A new green post-restore Step28 candidate has not been produced.
3. Universal any-motor energy gate is not closed.
4. Universal onboarding is not yet proven on a real identification-first flow.
5. Main orchestration scripts are still too monolithic.
6. Repository hygiene is not finished:
   - `README.md` still needs UTF-8 normalization and content cleanup.
   - root hygiene regression is still missing.
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

Remaining:
- add regression that root does not accumulate extra `PROJECT_MASTER_*` files again
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
  - checkpoint: `outputs/al31_rebuild_round1_20260318a/results_run/20260318_111437_env_research_al31_4_06kw_ai_id_ref/eval/actor_ep005.pth`
  - candidate: `al31_mid_04`
  - artifact: `outputs/al31_ep005_bridge_mid_20260322/al31_tuning_summary.json`
- exact current metrics:
  - `avg_power_saving_pct = 1.0951962740790293`
  - `avg_eta_gain_pct = 0.0005543431265536691`
  - `avg_power_saving_pct_min_seed = 0.23613917118256444`
  - `avg_eta_gain_pct_min_seed = -0.004466375337486284`
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
- current conclusion:
  - `AL31` is no longer missing a cheap reward-knob closure
  - even after the propagation fix, the current PPO local basin still prefers `actor_ep005 + al31_mid_04`

`AIR56`
- current best deploy pair:
  - checkpoint: `outputs/air56_ep003_tailfocus_micro2_20260322/results_run/20260322_131130_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/eval/actor_ep005.pth`
  - candidate: `mix04_base`
  - artifact: `outputs/air56_ep003_tailfocus_micro2_20260322/results_run/20260322_131130_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
- exact current metrics:
  - `avg_power_saving_pct = 0.6147884835796003`
  - `avg_eta_gain_pct = 0.0013968558560217836`
  - `avg_power_saving_pct_min_seed = 0.5151508570179902`
  - `avg_eta_gain_pct_min_seed = -0.019478794161115198`
  - `envelope_all_rows_pass = true`
  - `err_failures = 0.0`
- meaning:
  - mean power is green
  - mean eta is green
  - worst-case power is green
  - envelope is green
  - only remaining blocker is worst-case eta tail
- already proven dead ends:
  - candidate-only sweep around the best checkpoint
  - low-lr eta-edge warm-start with old hard reward gate
  - high-speed/high-load warm-start with old hard reward gate
  - longer eta-biased warm-start with old hard reward gate
  - high-speed scratch rebuild with old hard reward gate
  - running-eta reward micro-run
  - terminal-energy-bonus micro-run
  - running-eta reward with explicit `eta_episode_norm` observation
- current scientific conclusion:
  - `AIR56` is no longer a deploy-parameter problem
  - `AIR56` is now a reward/objective alignment problem
  - the current PPO micro-finetune regime still re-selects the old baseline checkpoint even after reward and feature extensions

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
    - external Step27 scan now accepts explicit `feature_keys` and can evaluate mixed feature-dimension checkpoint lineages
  - real validation after those fixes:
    - `AL31` true soft-gate rerun still kept `actor_ep_init` as best:
      - `outputs/al31_mid04_softgate_fixed_20260322/results_run/20260322_223527_tmp_al31_mid04_train_softgate_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - `AL31` terminal-bonus rerun still kept `actor_ep_init` as best:
      - `outputs/al31_mid04_terminal_fixed_20260322/results_run/20260322_224103_tmp_al31_mid04_train_terminal_20260322_ai_id_ref/external_step27_scan/al31_checkpoint_scan_summary.json`
    - `AIR56` terminal-bonus rerun still kept `actor_ep_init` as best:
      - `outputs/air56_ep005_terminal_fixed_20260322/results_run/20260322_224909_tmp_air56_ep022_mix04_train_terminal_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
    - `AIR56` running-eta reward plus explicit `eta_episode_norm` observation still kept `actor_ep_init` as best:
      - `outputs/air56_ep005_runningeta_obs_micro1_fix_20260322/results_run/20260322_230543_tmp_air56_ep022_mix04_train_20260322_ai_id_ref/external_step27_scan/air56_checkpoint_scan_summary.json`
  - meaning:
    - reward and feature extensions are now scientifically validated, not just implemented
    - they still do not beat the current strict baselines under low-compute micro-finetune conditions
    - therefore the remaining blocker is no longer “missing config plumbing” or “missing local gradient”
    - the blocker is the current PPO local basin / compute budget itself

#### W1.3 Immediate execution order
1. Stop reopening the current `AL31` and `AIR56` local basins with more micro-runs that only tweak reward scalars or one extra observation.
2. Keep the current factual baselines:
   - `AL31`: `actor_ep005 + al31_mid_04`
   - `AIR56`: `actor_ep005 + mix04_base`
3. Next justified W1 step is now materially different from all exhausted low-compute runs:
   - keep strict external Step27 selector
   - change the training basin, not just small scalar weights
   - acceptable next classes:
     - longer scratch training with the expanded energy objective / observation set already implemented
     - different hidden-size / policy-capacity regime for the MIC `id_ref` policy
     - if needed, a different optimizer / algorithmic family, not another micro-PPO replay of the same basin
4. Only after a materially different recipe closes both remaining tails, rerun full 3-motor live Step27 and then rebuild Step28 candidate.

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
   - `AL31`: `actor_ep005 + al31_mid_04`
   - `AIR56`: `actor_ep005 + mix04_base`
2. Do not spend more compute on the exhausted local recipes above.
3. Implement the next materially different training objective / policy-basin change.
4. Re-run `AL31` first only if that new recipe is ready; otherwise use `AIR56` as the primary closure target.
5. Only after both motors are green, rerun full Step27 and Step28.
6. After W1 closes, move to onboarding-energy closure, then refactor/docs/tests.
