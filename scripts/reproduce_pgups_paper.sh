#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-paper.txt

# Rebuild figures + data tables and export them into paper/pgups_2026/{fig,data}.
python tools/multi_motor_study_report.py --export-paper

# Optionally rebuild auxiliary paper tables derived from *local* training/sweep artifacts.
# On a clean clone these folders are not present (they are ignored by git), so we skip gracefully.
if [[ -d results_run ]]; then
  python tools/build_time_to_foc_summary_ru.py
fi
if [[ -d outputs/id_ref_sweep_pgups ]]; then
  python tools/build_id_ref_headroom_table.py
fi

# Build the learning figure (Fig. 6) from committed paper tables.
python tools/build_pgups_learning_figure.py

# Validate that published numbers match raw traces.
python tools/validate_pgups_study.py

# Build DOCX from Markdown (formulas are converted into Word equations when possible).
python tools/build_publication_from_markdown.py \
  --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md \
  --out-docx "paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx"
