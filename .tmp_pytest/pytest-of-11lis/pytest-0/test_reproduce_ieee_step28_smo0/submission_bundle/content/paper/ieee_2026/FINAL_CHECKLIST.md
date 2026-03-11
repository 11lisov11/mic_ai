# IEEE Final Checklist

Date: 2026-03-03

## Reproducibility
- [x] Frozen protocol script exists: `scripts/run_step28_ieee_protocol.ps1`
- [x] Frozen protocol script exists: `scripts/run_step28_ieee_protocol.sh`
- [x] Step28 packaging script exists: `scripts/package_ieee_step28.py`
- [x] One-command IEEE reproduce script exists: `tools/reproduce_ieee_step28.py`
- [x] IEEE promotion script exists: `tools/promote_ieee_release.py`
- [x] Submission lock script exists: `tools/freeze_ieee_submission_candidate.py`
- [x] Wrapper scripts exist: `scripts/reproduce_ieee_step28.ps1`, `scripts/reproduce_ieee_step28.sh`
- [x] Strict release wrappers exist: `scripts/release_ieee_submission_candidate.ps1`, `scripts/release_ieee_submission_candidate.sh`
- [x] Existing package verifier exists: `tools/verify_ieee_submission_candidate.py`
- [x] Verify wrappers exist: `scripts/verify_ieee_submission_candidate.ps1`, `scripts/verify_ieee_submission_candidate.sh`
- [x] Manuscript consistency checker exists: `tools/check_ieee_manuscript_consistency.py`
- [x] Manuscript template checker exists: `tools/check_ieee_manuscript_template.py`
- [x] Submission bundle builder exists: `tools/build_ieee_submission_bundle.py`
- [x] Release git planner exists: `tools/prepare_ieee_release_commit.py`
- [x] IEEE handoff note builder exists: `tools/build_ieee_submission_handoff.py`
- [x] Checkpoint registry is explicit: `config/checkpoint_registry.json`

## Pipeline smoke
- [x] Step27 smoke test in CI
- [x] Step28 summary smoke test in CI
- [x] Theory validation smoke test in CI
- [x] Frozen mini-baseline regression test (`benchmarks/baseline_summary_ci.json`)
- [x] Frozen IEEE candidate verify gate in CI (`ieee-frozen-verify`)
- [x] PowerShell wrapper verification in CI (`verify_ieee_submission_candidate.ps1`)
- [x] Manuscript consistency validation in verify/CI (`MANUSCRIPT_CONSISTENCY_REPORT`)
- [x] Manuscript template validation in verify/reproduce (`MANUSCRIPT_TEMPLATE_REPORT`)

## Core artifacts
- [x] `step27_per_seed_metrics.csv`
- [x] `step27_stats_motor_controller.csv`
- [x] `step27_final_pi_vs_foc_vs_mic.csv`
- [x] `step27_air56_acceptance.json`
- [x] `step27_reproducibility.json`
- [x] `step27_report.md`
- [x] `step28_ieee_summary.csv`
- [x] `step28_ieee_summary.md`
- [x] `derived_ieee/ieee_pi_foc_mic_stats.csv`
- [x] `derived_ieee/fig_ieee_pi_foc_mic_power.{png,pdf,svg}`
- [x] `derived_ieee/motor_tuning_acceptance_summary.csv`
- [x] `derived_ieee/motor_air56_tuning_report.md`
- [x] `derived_ieee/motor_al31_tuning_report.md`
- [x] `derived_ieee/motor_ao2_tuning_report.md`
- [x] `paper/ieee_2026/fig/fig2_pi_foc_mic_power.{png,pdf,svg}`
- [x] `paper/ieee_2026/fig/fig3_air56_working_characteristics.{png,pdf,svg}`
- [x] `paper/ieee_2026/data/release/<tag>/promotion_manifest.json`
- [x] `paper/ieee_2026/data/release/<tag>/release_snapshot.json`
- [x] `<step28_tag>/submission_candidate_lock.json` + lock check in `FINAL_CHECKLIST_AUTO.md`
- [x] `<step28_tag>/SUBMISSION_CANDIDATE.{md,json}` generated from lock + checklist
- [x] `<step28_tag>/RELEASE_COMMIT_MANIFEST.{md,json}` generated (immutable hash + git metadata)
- [x] `<step28_tag>/IEEE_SUBMISSION_DOSSIER.{md,json}` generated (single-file submission digest)
- [x] `submission_bundle/<tag>/submission_bundle_manifest.{md,json}` + `ieee_submission_<tag>.zip/.tar.gz`

## Physics validation
- [x] `tools/validate_theory_working_characteristics.py`
- [x] Bounds/shape checks for `eta` and `cosphi`
- [x] Against-passport table automation exists (`tools/build_against_passport_table.py`)
- [x] Passport script rejects invalid nominal rows and writes them to `failures` (instead of persisting `NaN` rows)
- [x] Current passport package has no hard failures (`failures=[]`)
- [~] AO2 is exported as `FOC` proxy point (`load_factor=0.8`) and MIC row is marked invalid in `warnings`

## Training quality gates (3 motors)
- [x] AIR56 acceptance in frozen package is satisfied (mean + worst-case).
- [x] AL31 is in positive zone with non-negative eta gain.
- [x] AO2 is out of negative saving zone (positive saving, non-negative eta gain).
- [x] Cross-motor guardrails are enforced in auto checklist (`motor_tuning_acceptance_summary.json` with thresholds).
- [x] Guardrails policy is versioned: `paper/ieee_2026/guardrails_policy.json`

## Current blocker to submission-quality claim
- [x] No critical blockers in current frozen package.
- [~] AO2 has small positive margin; monitor regressions in next retrain cycle.
