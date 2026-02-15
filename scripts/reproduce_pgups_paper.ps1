Param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python -m pip install -r requirements-paper.txt

# Rebuild figures + data tables and export them into paper/pgups_2026/{fig,data}.
& $Python tools/multi_motor_study_report.py --export-paper

# Optionally rebuild auxiliary paper tables derived from *local* training/sweep artifacts.
# On a clean clone these folders are not present (they are ignored by git), so we skip gracefully.
if (Test-Path results_run) {
  & $Python tools/build_time_to_foc_summary_ru.py
}
if (Test-Path outputs\\id_ref_sweep_pgups) {
  & $Python tools/build_id_ref_headroom_table.py
}

# Build the learning figure (Fig. 6) from committed paper tables.
& $Python tools/build_pgups_learning_figure.py

# Validate that published numbers match raw traces.
& $Python tools/validate_pgups_study.py

# Build DOCX from Markdown (formulas are converted into Word equations when possible).
& $Python tools/build_publication_from_markdown.py `
  --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md `
  --out-docx paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx
