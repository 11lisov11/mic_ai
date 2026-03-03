# IEEE Final Checklist

Date: 2026-03-03

## Reproducibility
- [x] Frozen protocol script exists: `scripts/run_step28_ieee_protocol.ps1`
- [x] Frozen protocol script exists: `scripts/run_step28_ieee_protocol.sh`
- [x] Step28 packaging script exists: `scripts/package_ieee_step28.py`
- [x] Checkpoint registry is explicit: `config/checkpoint_registry.json`

## Pipeline smoke
- [x] Step27 smoke test in CI
- [x] Step28 summary smoke test in CI
- [x] Theory validation smoke test in CI

## Core artifacts
- [x] `step27_per_seed_metrics.csv`
- [x] `step27_stats_motor_controller.csv`
- [x] `step27_final_pi_vs_foc_vs_mic.csv`
- [x] `step27_air56_acceptance.json`
- [x] `step27_reproducibility.json`
- [x] `step27_report.md`
- [x] `step28_ieee_summary.csv`
- [x] `step28_ieee_summary.md`

## Physics validation
- [x] `tools/validate_theory_working_characteristics.py`
- [x] Bounds/shape checks for `eta` and `cosphi`
- [~] Against-passport table automation exists (`tools/build_against_passport_table.py`)
- [ ] AO2 passport run is unstable (overflow at current research config)

## Training quality gates (3 motors)
- [~] AIR56 `start_stop` improved, but full acceptance not reached
- [~] AL31 in positive zone, needs final lock
- [~] AO2 near-neutral with conservative profile, needs eta >= 0 lock

## Current blocker to submission-quality claim
- [ ] AIR56 acceptance gate (`avg_power_saving > 0.5`, `avg_eta_gain >= 0`) is not yet satisfied.
