# Статья для «Известия ПГУПС» (2026)

Содержимое:

- `article_mic_ieee_vak_pgups.md` - исходник статьи (Markdown).
- `fig/` - рисунки для статьи.
- `СТАТЬЯ_MIC_ПГУПС_2026.docx` - собранная версия (Word).
- `pgups_requirements_check_20260212.md` - чек-лист соответствия требованиям.

## Пересборка DOCX

```bash
python -m pip install -r requirements-paper.txt
python tools/build_publication_from_markdown.py --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md --out-docx paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx
```
