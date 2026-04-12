# IEEE 2026 Package

This folder stores publication-ready artifacts for the IEEE submission workflow.

## Latest strict-verified packages

- current canonical 3-motor release:
  - `paper/ieee_2026/data/step28/20260412_postrestore_ai_3motors_release`
- historical 2-motor release milestone:
  - `paper/ieee_2026/data/step28/20260412_postrestore_ai_2motors_release`

Core structure:
- `paper/ieee_2026/manuscript.md` - manuscript scaffold synchronized with frozen step28 packages.
- `paper/ieee_2026/fig/` - promoted publication figures (see `paper/ieee_2026/fig/README.md`).
- `paper/ieee_2026/data/step28/<tag>/` - frozen reproducibility packages.

## Step28 package flow

One-command reproducibility (recommended):

```bash
python tools/reproduce_ieee_step28.py \
  --out-root outputs/progress_step28_ieee_repro \
  --package-root paper/ieee_2026/data/step28 \
  --package-tag 20260303_repro \
  --mic-mode rule
```

Windows PowerShell:

```powershell
.\scripts\reproduce_ieee_step28.ps1 --out-root outputs/progress_step28_ieee_repro --package-tag 20260303_repro --mic-mode rule
```

Strict submission-candidate wrapper (promotion + lock strict + policy):

```powershell
.\scripts\release_ieee_submission_candidate.ps1 -OutRoot outputs/release_ieee_submission_candidate -Tag 20260303_candidate -MicMode rule
```

Linux wrapper:

```bash
./scripts/release_ieee_submission_candidate.sh outputs/release_ieee_submission_candidate 20260303_candidate rule
```

For `tools/reproduce_ieee_step28.py`, verification is non-strict by default (to support smoke packages).  
Use `--strict-verify` for release-grade runs.

Verification of an existing frozen package (without rerunning step27):

```bash
python tools/verify_ieee_submission_candidate.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --guardrails-policy paper/ieee_2026/guardrails_policy.json \
  --allow-dirty \
  --strict
```

PowerShell wrapper:

```powershell
.\scripts\verify_ieee_submission_candidate.ps1 -Step28Dir paper/ieee_2026/data/step28/<tag> -AllowDirty
```

Linux wrapper:

```bash
./scripts/verify_ieee_submission_candidate.sh paper/ieee_2026/data/step28/<tag> paper/ieee_2026 paper/ieee_2026/guardrails_policy.json paper/ieee_2026/manuscript.md
```

Build final submission bundle archive (zip + tar.gz + hash manifest):

```bash
python tools/build_ieee_submission_bundle.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag> \
  --strict
```

Prepare git release commit plan (and optionally apply/tag/push):

```bash
python tools/prepare_ieee_release_commit.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag>
```

Windows wrapper:

```powershell
.\scripts\prepare_ieee_release_commit.ps1 -Step28Dir paper/ieee_2026/data/step28/<tag> -Tag <tag>
```

Build IEEE handoff note (portal upload checklist):

```bash
python tools/build_ieee_submission_handoff.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag> \
  --strict
```

Run frozen summary drift guard against baseline:

```bash
python tools/check_step28_summary_regression.py \
  --summary-csv paper/ieee_2026/data/step28/<tag>/step28_ieee_summary.csv \
  --baseline-json benchmarks/step28_ieee_summary_baseline_20260303_ai_config_locked_nodrift.json \
  --strict
```

Build frozen release notes:

```bash
python tools/build_ieee_release_notes.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag> \
  --strict
```

Build camera-ready checklist:

```bash
python tools/build_ieee_camera_ready_checklist.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag> \
  --strict
```

Build rebuttal evidence pack (tables/figures/hashes/logs):

```bash
python tools/build_ieee_rebuttal_evidence_pack.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --ieee-root paper/ieee_2026 \
  --tag <tag> \
  --strict
```

Note: wrappers assume Python environment with project dependencies installed (`pip install -r requirements.txt`).

This command runs mode1/mode2 step27, builds step28 summary, packages publication artifacts, builds IEEE-derived tables/figures (`derived_ieee/`), builds per-motor tuning acceptance reports, freezes `submission_candidate_lock.json` (SHA lockfile), generates `FINAL_CHECKLIST_AUTO.md`, writes `SUBMISSION_CANDIDATE.{md,json}`, validates manuscript consistency + template, verifies frozen candidate (`VERIFY_SUBMISSION_CANDIDATE.json`), builds submission bundle archive, and writes `step28_reproduce_manifest.json`.

If you want to run promotion in the same pass:

```bash
python tools/reproduce_ieee_step28.py \
  --out-root outputs/progress_step28_ieee_repro \
  --package-root paper/ieee_2026/data/step28 \
  --package-tag 20260303_repro \
  --mic-mode rule \
  --promote-release \
  --ieee-root paper/ieee_2026 \
  --pgups-fig-dir paper/pgups_2026/fig
```

Optional promotion to manuscript-ready release:

```bash
python tools/promote_ieee_release.py \
  --step28-dir paper/ieee_2026/data/step28/20260303_repro \
  --ieee-root paper/ieee_2026 \
  --pgups-fig-dir paper/pgups_2026/fig \
  --tag 20260303_repro
```

This copies canonical figures into `paper/ieee_2026/fig/` and writes a release snapshot under `paper/ieee_2026/data/release/<tag>/`.

To require publication/release assets in the lock (strict submission freeze), use:

```bash
python tools/reproduce_ieee_step28.py \
  --out-root outputs/progress_step28_ieee_repro \
  --package-root paper/ieee_2026/data/step28 \
  --package-tag 20260303_repro \
  --mic-mode rule \
  --promote-release \
  --freeze-require-publication-assets \
  --freeze-require-release-assets
```

Auto checklist now includes cross-motor guardrails from `derived_ieee/motor_tuning_acceptance_summary.json` with default thresholds:
- `air56 >= 0.5%`
- `al31 >= 0.0%`
- `ao2 >= 0.05%`

These are release-level power-saving guardrails. Scenario-level `eta/current/start_stop`
constraints are enforced earlier by the canonical Step27 acceptance envelopes and should
not be reinterpreted here as a second hidden aggregate `eta >= 0` gate for generic motors.

Policy source (versioned):
- `paper/ieee_2026/guardrails_policy.json`

Override if needed:

```bash
python tools/build_ieee_final_checklist.py \
  --step28-dir paper/ieee_2026/data/step28/<tag> \
  --guardrails-policy paper/ieee_2026/guardrails_policy.json \
  --motor-saving-thresholds "air56:0.5,al31:0.0,ao2:0.05"
```

---

1. Run the frozen protocol:

```bash
scripts/run_step28_ieee_protocol.sh outputs/progress_step28_ieee_ai
```

or on Windows:

```powershell
.\scripts\run_step28_ieee_protocol.ps1 -OutRoot outputs/progress_step28_ieee_ai -MicMode ai
```

2. Package only publication artifacts (without heavy raw traces):

```bash
python scripts/package_ieee_step28.py \
  --step28-out outputs/progress_step28_ieee_ai \
  --dest-root paper/ieee_2026/data/step28 \
  --theory-csv paper/pgups_2026/fig/working_characteristics_air56_foc_mic_table.csv \
  --passport-dir paper/ieee_2026/data/passport/<tag> \
  --strict
```

3. Result:
- `paper/ieee_2026/data/step28/<tag>/step28_ieee_summary.csv`
- `paper/ieee_2026/data/step28/<tag>/step28_ieee_summary.md`
- mode-specific `step27_*` tables and reports
- `package_manifest.json`
- `submission_candidate_lock.json`
- `SUBMISSION_CANDIDATE.md`
- `SUBMISSION_CANDIDATE.json`
- `RELEASE_COMMIT_MANIFEST.md`
- `RELEASE_COMMIT_MANIFEST.json`
- `IEEE_SUBMISSION_DOSSIER.md`
- `IEEE_SUBMISSION_DOSSIER.json`
- `MANUSCRIPT_TEMPLATE_REPORT.md`
- `MANUSCRIPT_TEMPLATE_REPORT.json`
- `VERIFY_SUBMISSION_CANDIDATE.json`
- `submission_bundle/<tag>/submission_bundle_manifest.{md,json}`
