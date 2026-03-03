# IEEE 2026 Package

This folder stores publication-ready artifacts for the IEEE submission workflow.

## Step28 package flow

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
