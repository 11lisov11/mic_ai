# Структура проекта (GitHub)

Дата: 2026-02-13

## 1) Что публикуется в репозитории

- Исходный код: `mic_ai/`, `models/`, `control/`, `simulation/`, `drivers/`, `metrics/`, `tools/`.
- Конфигурации экспериментов: `config/`.
- Тесты и CI: `tests/`, `.github/workflows/`.
- Материалы статьи (Markdown, DOCX, рисунки): `paper/pgups_2026/`.

## 2) Что хранится локально и не попадает в git

Эти каталоги игнорируются в `.gitignore`, чтобы не публиковать большие артефакты:

- `outputs/` - результаты прогонов, рисунки, таблицы, CSV и др.
- `results_run/` - прогоны обучения и чекпойнты.
- `archive/` - локальный архив/легаси.
- `docs/legacy_examples/` - крупные примеры/шаблоны, не относящиеся к коду.

## 3) Воспроизводимость

Быстрый прогон тестов:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Пересборка DOCX статьи (опционально):

```bash
python -m pip install -r requirements-paper.txt
python tools/build_publication_from_markdown.py --src-md paper/pgups_2026/article_mic_ieee_vak_pgups.md --out-docx paper/pgups_2026/СТАТЬЯ_MIC_ПГУПС_2026.docx
```
