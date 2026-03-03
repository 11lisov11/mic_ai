# PROJECT MASTER PLAN STATUS (2026-03-03)

This file tracks factual execution status for `PROJECT_MASTER_PLAN_IEEE_3MOTORS_20260303.md`.

## 10.1 Pipeline/Infra
- [x] `step27_pipeline.py` API compatibility restored.
- [x] `ai_eval_*` parameters added for all 3 research configs.
- [x] Checkpoint registry added (`config/checkpoint_registry.json`) with step27/tune fallback.
- [x] Step27 smoke is in CI (`step27-smoke` job).
- [x] Step28 smoke is in CI (`step28-summary-smoke` job).

## 10.2 Theory/Physics
- [x] `tools/validate_theory_working_characteristics.py` implemented.
- [x] Shape validation is included for `M2/I1/n2/eta/cosphi`.
- [~] Against-passport table is generated, but AO2 still has overflow in passport run
  (`paper/ieee_2026/data/passport/20260303_config_locked/passport_compare_3motors.json` contains failure diagnostics).
- [~] Validation report integrated into publication flow partially
  (theory smoke exists in CI; publication packaging script added).

## 10.3 Training
- [x] Unified 3-motor pipeline exists (`tools/train_3motors_pipeline.py`).
- [x] Joint training mode exists (`joint-domain-randomized`) with fine-tune mode.
- [x] AIR56 `start_stop` target-tuning is locked for fixed-seed protocol:
  `outputs/progress_step27_ai_config_locked_20260303/step27_air56_acceptance.json` -> `mean_pass=true`, `worst_case_pass=true`.
- [x] AL31 stabilized with tuned safe profile (`manual_safe_01`) and non-negative eta.
- [x] AO2 moved to non-negative eta on fixed-seed protocol (`avg_eta_gain_pct_mean=+0.019%`).
- [~] Robustness under plant perturbation (`seed_perturb_level=0.2`) is still open:
  AL31 and AIR56 regress in `outputs/progress_step28_ieee_ai_config_locked_20260303`.

## 10.4 Tests/CI
- [x] Unit tests for metrics expanded (power factor / checkpoint registry / packaging).
- [x] Integration smoke tests for step27/step28 present and passing.
- [x] Regression reproducibility checks exist (`tests/test_reproducibility_hash.py`).
- [x] CI gates extended (step27/step28/theory smoke jobs).

## 10.5 IEEE package
- [x] `paper/ieee_2026/` created.
- [ ] Final IEEE figures/tables still pending.
- [x] One-command reproducibility scripts exist (`run_step28_ieee_protocol.ps1/.sh`).
- [x] Packaging script is used to produce current IEEE dataset bundle:
  `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift`.

## Latest package artifact
- `paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift/`
