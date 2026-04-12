# PROJECT MASTER PLAN (ACTIVE)

Date updated: `2026-04-12`  
Repository: `C:\mic_theory`

## Status

The project is complete on the full `3-motor` scope:

- `AIR56`
- `AL31`
- `AO2`

Canonical strict-verified release:

- [20260412_postrestore_ai_3motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release)
- [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)
- `verification_ok = true`

Historical strict-verified `2-motor` release kept for provenance:

- [20260412_postrestore_ai_2motors_release](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release)

## Canonical Completion Criteria

The project counts as `100% done` only when all items below are true.

1. Strict `Step27/Step28` package exists for all three motors.
2. `VERIFY_SUBMISSION_CANDIDATE.json` is green.
3. Per-motor acceptance is green for `AIR56`, `AL31`, and `AO2`.
4. Root documentation reflects the real scope and active release tag.
5. Regression tests are green after the final code changes.

Status on `2026-04-12`:

- all five conditions are satisfied

## Final Canonical Artifacts

### Release package

- [FINAL_CHECKLIST_AUTO.md](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/FINAL_CHECKLIST_AUTO.md)
- [SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/SUBMISSION_CANDIDATE.json)
- [IEEE_SUBMISSION_DOSSIER.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/IEEE_SUBMISSION_DOSSIER.json)
- [VERIFY_SUBMISSION_CANDIDATE.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/VERIFY_SUBMISSION_CANDIDATE.json)

### Per-motor acceptance

- [motor_tuning_acceptance_summary.json](C:/mic_theory/paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release/derived_ieee/motor_tuning_acceptance_summary.json)

Confirmed final state:

- `AIR56`: `acceptance_pass = true`
- `AL31`: `acceptance_pass = true`
- `AO2`: `acceptance_pass = true`

### Reproduce outputs

- [step28_reproduce_manifest.json](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/step28_reproduce_manifest.json)
- [mode1 summary](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/mode1_foc_encoder_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv)
- [mode2 summary](C:/mic_theory/outputs/reproduce_ieee_step28_20260412_3motor/mode2_foc_sensorless_vs_mic_sensorless/step27_final_pi_vs_foc_vs_mic.csv)

## AO2 Closure Record

`AO2` was the last real blocker.

The closure path that worked:

1. Detect the mismatch between the old live runtime config and the motor nameplate.
2. Build a nameplate-first AO2 branch.
3. Confirm that the remaining blocker is `FOC` headroom / saturation, not only PPO quality.
4. Add optional `field_weakening` to the FOC controller.
5. Sweep cheap `field_weakening` parameters on the already tuned AO2 actor.
6. Promote the strict-green configuration into the live AO2 research config and registry.
7. Rebuild the full strict `3-motor` package.

Canonical AO2 artifacts:

- diagnosis tool: [diagnose_motor_nominal_consistency.py](C:/mic_theory/tools/diagnose_motor_nominal_consistency.py)
- preserved nameplate-first config: [env_backlog_ao2_nameplate_first.py](C:/mic_theory/config/env_backlog_ao2_nameplate_first.py)
- tuned AO2 backlog config: [env_backlog_ao2_nameplate_foc_tuned.py](C:/mic_theory/config/env_backlog_ao2_nameplate_foc_tuned.py)
- live AO2 config: [env_research_ao2_32_4_3kw.py](C:/mic_theory/config/env_research_ao2_32_4_3kw.py)
- canonical strict-green FW sweep result: [fw_c summary](C:/mic_theory/outputs/ao2_fw_grid_20260412af/fw_c/ao2_checkpoint_scan_summary.json)

## Final Regression Reference

Focused final regression after the AO2 closure work:

- `python -m pytest -q tests/test_step27_report_markdown.py tests/test_step27_hybrid_trigger.py tests/test_vector_foc_field_weakening.py tests/test_scan_step27_checkpoints.py tests/test_train_ai_id_ref_external_step27.py tests/test_diagnose_motor_nominal_consistency.py`
- `60 passed`

Full repository regression must remain green before final push:

- `python -m pytest -q`

## Guardrails

- Do not weaken acceptance thresholds to keep the package green.
- Do not remove the preserved AO2 diagnosis/tuning artifacts.
- Do not create a new root plan while this file is current.
- Do not treat temporary probe configs under `outputs/` as canonical live configs.
- The canonical AO2 live path is:
  - [env_research_ao2_32_4_3kw.py](C:/mic_theory/config/env_research_ao2_32_4_3kw.py)
  - [checkpoint_registry.json](C:/mic_theory/config/checkpoint_registry.json)

## Optional Backlog

Nothing below is required for project completion anymore.

1. Expand universal onboarding proofs from the current release slice to a broader AO2-specific productization path.

Completed on `2026-04-12` after main project closure:

- monolithic `tools/reproduce_ieee_step28.py` was split into path/command builders without behavior change
- `step27_air56_acceptance.json` now has a backward-compatible generic successor:
  - `step27_motor_acceptance.json`
  - packaging/checklist/freeze/summary tools accept both names
  - new packages emit the generic file while preserving the legacy alias
