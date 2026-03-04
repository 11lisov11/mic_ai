# 3-Motor Operations Runbook (AIR56/AL31/AO2)

## Scope
This runbook defines a reproducible operational flow for:
- baseline evaluation (`step27`);
- IEEE package reproduction (`step28`);
- robust hardening sweep (AL31/AO2);
- 3-motor training pipeline with `eval-first` policy.

The goal is to avoid ad-hoc runs and keep traceability to machine-readable artifacts.

## 1) Baseline Step27
Run one deterministic baseline for selected motors/seeds/scenarios:

```bash
python tools/step27_pipeline.py \
  --motors air56,al31,ao2 \
  --seeds 101,202,303,404,505 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --checkpoint-registry config/checkpoint_registry.json \
  --seed-perturbation --seed-perturb-level 0.2 \
  --out-dir outputs/step27_baseline_YYYYMMDD
```

Required artifacts:
- `step27_per_seed_metrics.csv`
- `step27_stats_motor_controller.csv`
- `step27_final_pi_vs_foc_vs_mic.csv`
- `step27_reproducibility.json`

## 2) IEEE Step28 Reproduction

```bash
python tools/reproduce_ieee_step28.py \
  --out-root outputs/step28_YYYYMMDD \
  --package-root paper/ieee_2026/data/step28 \
  --package-tag TAG_YYYYMMDD \
  --motors air56,al31,ao2 \
  --seeds 101,202,303,404,505 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --checkpoint-registry config/checkpoint_registry.json
```

Required artifacts:
- `step28_ieee_summary.csv`
- `package_manifest.json`
- `derived_ieee/ieee_pi_foc_mic_stats.csv`
- `VERIFY_SUBMISSION_CANDIDATE.json`

## 3) Robust Hardening (No Retrain)

```bash
python tools/robust_motor_hardening.py \
  --motors al31,ao2 \
  --seeds 101,202,303 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --perturb-levels 0.0,0.1,0.2,0.3,0.4 \
  --out-dir outputs/robust_hardening_YYYYMMDD
```

Consolidate all hardening runs:

```bash
python tools/build_robust_hardening_consolidated.py \
  --root outputs \
  --out-dir outputs/robust_hardening_consolidated_YYYYMMDD
```

## 4) Training Pipeline (Eval-First, No Unnecessary Retrain)
Initial run:

```bash
python tools/train_3motors_pipeline.py \
  --mode joint-domain-randomized \
  --motors air56,al31,ao2 \
  --seeds 101,202,303 \
  --joint-cycles 2 \
  --joint-cycle-episodes 40 \
  --out-dir outputs/train3_YYYYMMDD
```

Follow-up run with reuse:

```bash
python tools/train_3motors_pipeline.py \
  --mode joint-domain-randomized \
  --motors air56,al31,ao2 \
  --seeds 101,202,303 \
  --resume-manifest <path/to/training_manifest_3motors.json> \
  --eval-first \
  --out-dir outputs/train3_eval_first_YYYYMMDD
```

Expected run artifacts:
- `training_manifest_3motors.json`
- `training_protocol_3motors.json`
- `training_repro_package_3motors.json`
- `checkpoints_registry_3motors.json`

## 5) Acceptance Protocol
Run checks in this exact order:
1. CSV/JSON contracts:
```bash
python -m pytest -q tests/test_artifact_contracts.py
```
2. Theory checks:
```bash
python tools/build_theory_validation_reports.py \
  --step28-tag <STEP28_TAG> \
  --passport-root paper/ieee_2026/data/passport \
  --out-root paper/ieee_2026/data/theory_validation
```
3. Scenario envelopes:
```bash
python tools/check_motor_acceptance_envelopes.py \
  --runs-root <step27_or_extended_runs_root> \
  --out-dir <out_dir>/acceptance_envelopes
```
4. Release verification:
```bash
python tools/verify_ieee_submission_candidate.py \
  --step28-dir paper/ieee_2026/data/step28/<TAG> \
  --ieee-root paper/ieee_2026
```

`hard-ready` is allowed only if all four checks pass.

## 6) Cross-Motor Generalization (Held-Out)
Evaluate transfer from source motors to held-out targets:

```bash
python tools/eval_cross_motor_generalization.py \
  --mode heldout \
  --source-motors air56,al31 \
  --target-motors ao2 \
  --seeds 101,202,303 \
  --scenarios speed_step,ramp,load_step,start_stop \
  --checkpoint-registry config/checkpoint_registry.json \
  --out-dir outputs/cross_motor_generalization_YYYYMMDD
```

Artifacts:
- `cross_motor_generalization_per_seed.csv`
- `cross_motor_generalization_summary.csv`
- `cross_motor_generalization_gap_vs_native.csv`
- `cross_motor_generalization_report.md`

## 7) Troubleshooting
- `verify_ieee_submission_candidate` fails with `dossier_ok=false`:
  - Rebuild summary and derived tables from the same step28 tag.
  - Ensure `manuscript.md` references existing figures/tables.
- `theory_validation_summary.json` has `all_passed=false`:
  - Inspect `*_theory_validation_report.json` per motor.
  - Re-check smoothing/filtering in working-characteristics preprocessing.
- `eval-first` does not reuse runs:
  - Verify `--resume-manifest` points to a manifest with `training_acceptance_matrix_3motors.csv`.
  - Ensure checkpoint and episodes files still exist.
- AO2 envelope fails:
  - Re-run robust hardening with wider stage1 and tighter safe constraints.
  - Do not relax physics sanity checks for eta/cosphi/current.

## 8) Integration Contour
Use single command orchestration:

```bash
python tools/run_integration_pipeline.py \
  --out-root outputs/integration_pipeline \
  --motors air56 \
  --seeds 101 \
  --scenarios speed_step
```

Report outputs:
- `integration_pipeline_report.json`
- `integration_pipeline_report.md`
