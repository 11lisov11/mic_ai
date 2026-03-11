# IEEE Final Checklist (Auto)

- step28_dir: `C:\mic_theory\paper\ieee_2026\data\step28\20260308_postrestore_ai`

## Core artifacts
- [x] `step28_ieee_summary.csv`
- [x] `step28_ieee_summary.md`
- [x] `package_manifest.json`

## Mode artifacts
### mode1_foc_encoder_vs_mic_sensorless
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_per_seed_metrics.csv`
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_stats_motor_controller.csv`
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv`
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_air56_acceptance.json`
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_reproducibility.json`
- [x] `mode1_foc_encoder_vs_mic_sensorless/step27_report.md`
- [ ] AIR56 acceptance: mean_pass=False, worst_case_pass=False
- [x] reproducibility: stable_vs_previous=None, sha=b975f3d7ff145b7ebade3db54edb0b84b8f162c126d6a924c28d4f808860eea2

### mode2_foc_sensorless_vs_mic_sensorless
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_per_seed_metrics.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_stats_motor_controller.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_air56_acceptance.json`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_reproducibility.json`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_report.md`
- [ ] AIR56 acceptance: mean_pass=False, worst_case_pass=False
- [x] reproducibility: stable_vs_previous=None, sha=b975f3d7ff145b7ebade3db54edb0b84b8f162c126d6a924c28d4f808860eea2

## Derived IEEE figures/tables
- [x] `derived_ieee/ieee_pi_foc_mic_stats.csv`
- [x] `derived_ieee/ieee_pi_foc_mic_stats.md`
- [x] `derived_ieee/fig_ieee_pi_foc_mic_power.png`
- [x] `derived_ieee/fig_ieee_pi_foc_mic_power.pdf`
- [x] `derived_ieee/fig_ieee_pi_foc_mic_power.svg`

## Motor acceptance guardrails
- policy: `C:\mic_theory\paper\ieee_2026\guardrails_policy.json`
- thresholds: `air56>=+0.500%, al31>=+0.000%, ao2>=+0.050%`
- [x] `derived_ieee/motor_tuning_acceptance_summary.json`
- [ ] motor=air56 acceptance_pass=False saving_mean=+0.304% saving_min=+0.104% threshold=+0.500%
- [ ] motor=al31 acceptance_pass=False saving_mean=+0.134% saving_min=-0.486% threshold=+0.000%
- [ ] motor=ao2 acceptance_pass=False saving_mean=+8.520% saving_min=+0.839% threshold=+0.050%

## Passport
- [ ] `passport/passport_compare_3motors.(csv|md|json)`
- [x] passport checks skipped (artifacts missing)

## Publication assets
- [x] `manuscript.md`
- [x] `fig/fig1_mic_methodology.png`
- [x] `fig/fig2_pi_foc_mic_power.pdf`
- [x] `fig/fig3_air56_working_characteristics.pdf`
- [x] `fig/fig4_cross_motor_robustness.pdf`
- [x] `fig/fig5_training_to_foc.pdf`

## Submission lock
- [x] `submission_candidate_lock.json`
- [x] lock_ok=True, required_files_missing=0

## Submission readiness
- [ ] ready_for_submission: `False`

### Blocking items
- AIR56 acceptance gate is not fully satisfied
- motor acceptance guardrails failed
