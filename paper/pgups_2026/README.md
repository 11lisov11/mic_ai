# Статья для «Известия ПГУПС» (2026)

Содержимое:

- `article_mic_ieee_vak_pgups.md` - исходник статьи (Markdown).
- `fig/` - рисунки для статьи.
- `СТАТЬЯ_MIC_ПГУПС_2026.docx` - собранная версия (Word).
- `pgups_requirements_check_20260212.md` - чек-лист соответствия требованиям.
- `data/` - трассы и таблицы метрик для воспроизведения результатов статьи без RL-чекпойнтов.

## Пересборка DOCX

```bash
python -m pip install -r requirements-paper.txt
python tools/build_publication_from_markdown.py --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md --out-docx paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx
```

## Воспроизведение рисунков и чисел

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-paper.txt
python tools/multi_motor_study_report.py --export-paper
python tools/build_pgups_learning_figure.py
python tools/validate_pgups_study.py
```

Скрипт (Windows): `scripts/reproduce_pgups_paper.ps1`.
