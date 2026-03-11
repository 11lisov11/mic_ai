# RCA: Unexpected Workspace Wipe (`C:\mic_theory`)

Date: 2026-03-08  
Scope: recovery + root-cause analysis after repeated full workspace deletion.

## Recovery
- Workspace restored from remote backup repository:
  - `https://github.com/11lisov11/mic_ai.git`
  - branch: `main`
  - restored commit at recovery time: `6d5cf9e`

## Forensic Findings
- Static code audit across repository found only these recursive delete operations:
  - `tools/build_ieee_submission_bundle.py` -> `shutil.rmtree(payload_root)`
  - `tools/update_study_final_from_run.py` -> `shutil.rmtree(backup)`
- Point deletions (`unlink`) exist only for explicit single files:
  - `mic_ai/tools/drive_characteristics_ai.py`
  - `tools/build_air56_working_characteristics_article.py`
- `tools/robust_motor_hardening.py` (the command active during incident) had **no delete operation**.

## Risk Identified
- Recursive deletions above had no hard guardrails, so an incorrect path argument could remove unexpected directories.
- This was a latent safety flaw even if it is not proven to be the direct trigger of the wipe.

## Hardening Applied
- Added guarded deletion helper:
  - `tools/common_utils.py` -> `safe_rmtree(...)`
  - checks:
    - target is a directory;
    - target is inside repository root;
    - target is not repository root itself;
    - minimum relative depth;
    - optional allowed leaf-name allowlist.
- Switched deletion call sites to guarded delete:
  - `tools/build_ieee_submission_bundle.py`
  - `tools/update_study_final_from_run.py`
- Added output-path guard to prevent broad/unsafe run directories:
  - `tools/robust_motor_hardening.py` -> `_validate_out_dir(...)`
  - rejects repo root, non-repo paths, and non-`outputs/...` targets.

## Validation
- Syntax checks:
  - `python -m py_compile tools/common_utils.py tools/build_ieee_submission_bundle.py tools/update_study_final_from_run.py tools/robust_motor_hardening.py`
- Smoke tests:
  - `pytest -q tests/test_freeze_ieee_submission_candidate_smoke.py` -> passed.
  - `python tools/robust_motor_hardening.py ... --dry-run --out-dir outputs/_guard_test` -> passed.
  - `python tools/robust_motor_hardening.py ... --dry-run --out-dir .` -> fails by design with guard error.

## Current Conclusion
- Direct code path that explicitly deletes `C:\mic_theory` was not found.
- Unsafe recursive-delete primitives existed and are now guarded.
- If full workspace wipe repeats after these guardrails, next step is OS-level file operation auditing
  (process monitor / Windows audit) to capture exact deleting process.
