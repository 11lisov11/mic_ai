# Any-Motor Onboarding

## Goal
Train MIC AI for a new motor from passport data (and optional identification data), then validate the trained policy on benchmark motors without retraining those benchmarks.

Active default benchmark scope:
- `air56`
- `al31`

`AO2` is preserved as backlog research and can still be included explicitly with:
```bash
--benchmark-motors air56,al31,ao2
```

## One-command flow
```bash
python tools/train_any_motor_pipeline.py \
  --passport-json path/to/new_motor_passport.json \
  --motor-key new_motor \
  --out-dir outputs/train_any_motor_pipeline
```

Pipeline now includes:
- optional multi-attempt training (`--max-train-attempts`);
- benchmark parameter search (`id_ref_alpha`, `delta_id_max`) without retraining;
- acceptance gate on benchmark metrics (enabled by default).
- default benchmark list aligned with the active 2-motor release scope.

## Input passport JSON
Minimal required fields:
- `P_n` (W) or `P_kW`
- `U_ll`
- `I_n`

Recommended fields:
- `cos_phi_n`
- `eta_n`
- `f_n`
- `p`
- `n_rated`
- `connection` (`Y` or `D`)
- `J`

## Optional identification
You can provide either:
1. Precomputed identification JSON:
```bash
--ident-json path/to/ident_result.json
```
2. Raw identification test datasets (all three are required together):
```bash
--ident-rs-leq path/to/rs_leq.json \
--ident-locked-rotor-q path/to/locked_rotor_q.json \
--ident-mech-runup path/to/mech_runup.json
```

## Dry-run
```bash
python tools/train_any_motor_pipeline.py \
  --passport-json path/to/new_motor_passport.json \
  --motor-key new_motor \
  --dry-run \
  --run-tag smoke
```

## Acceptance and auto-tuning
Default acceptance gate:
- `err_ok_rate >= 1.0` for each benchmark motor;
- all benchmark motors must pass.

Optional energy gate (disabled by default):
- `--accept-power-saving-mean-min <value>` enables threshold on `power_saving_pct_mean`.

For the active 2-motor release scope, the default onboarding gate is evaluated on `air56,al31`.
To re-open `AO2` research during onboarding validation, pass `--benchmark-motors air56,al31,ao2`.

You can tune thresholds and search grid:
```bash
python tools/train_any_motor_pipeline.py \
  --passport-json path/to/new_motor_passport.json \
  --motor-key new_motor \
  --max-train-attempts 3 \
  --train-episodes-scale 1.5 \
  --benchmark-search-alpha-grid 0.6,0.8,1.0 \
  --benchmark-search-delta-grid 0.15,0.25,0.35 \
  --accept-err-ok-rate-min 1.0 \
  --accept-power-saving-mean-min 0.0
```

Disable hard acceptance gate (not recommended):
```bash
--no-acceptance-gate
```

Re-validate an existing policy without retraining:
```bash
python tools/train_any_motor_pipeline.py \
  --passport-json path/to/new_motor_passport.json \
  --motor-key new_motor \
  --skip-training \
  --init-checkpoint path/to/best_actor.pth \
  --benchmark-search-alpha-grid 0.6,0.8,1.0 \
  --benchmark-search-delta-grid 0.1,0.2,0.3
```

## Main artifacts
- `any_motor_onboarding_report.json`
- `any_motor_onboarding_report.md`
- `normalized_passport.json`
- `motor_params_final.json`
- generated config: `config/generated/env_onboard_<motor_key>.py`
- benchmark validation:
- `benchmark_validation_plan.json`
- `benchmark_validation_summary.json`
- `benchmark_search_summary.json`
- `training_attempts.json`
