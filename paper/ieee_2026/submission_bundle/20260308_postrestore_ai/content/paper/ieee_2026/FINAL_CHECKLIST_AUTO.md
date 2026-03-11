# IEEE Final Checklist (Auto)

- step28_dir: `C:\mic_theory\paper\ieee_2026\data\step28\20260303_ai_config_locked_nodrift`

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
- [x] AIR56 acceptance: mean_pass=True, worst_case_pass=True
- [x] reproducibility: stable_vs_previous=None, sha=e74b3a5a14cba32dec8f3f7d17413a1df57dba17f5d9371d3968c3040ba90154

### mode2_foc_sensorless_vs_mic_sensorless
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_per_seed_metrics.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_stats_motor_controller.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_air56_acceptance.json`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_reproducibility.json`
- [x] `mode2_foc_sensorless_vs_mic_sensorless/step27_report.md`
- [x] AIR56 acceptance: mean_pass=True, worst_case_pass=True
- [x] reproducibility: stable_vs_previous=None, sha=e74b3a5a14cba32dec8f3f7d17413a1df57dba17f5d9371d3968c3040ba90154

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
- [x] motor=air56 acceptance_pass=True saving_mean=+0.576% saving_min=+0.576% threshold=+0.500%
- [x] motor=al31 acceptance_pass=True saving_mean=+2.085% saving_min=+2.085% threshold=+0.000%
- [x] motor=ao2 acceptance_pass=True saving_mean=+0.092% saving_min=+0.092% threshold=+0.050%

## Passport
- [x] `passport/passport_compare_3motors.(csv|md|json)`
- [x] passport failures: 0
- [x] passport warnings: 2

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
- [x] ready_for_submission: `True`
