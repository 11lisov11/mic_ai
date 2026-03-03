# UTF-8 Policy

This repository uses UTF-8 (without BOM) as the canonical encoding for source code, configs, docs and CSV/JSON metadata.

## Rules

1. New text files must be saved as UTF-8.
2. Python I/O must use explicit `encoding="utf-8"` unless there is a strict external requirement.
3. Mixing cp1251/ANSI and UTF-8 in the same workflow is not allowed.
4. If a legacy file with broken encoding is detected, convert it to UTF-8 in a separate commit.

## Practical checks

1. Prefer `read_text(encoding="utf-8")` / `write_text(..., encoding="utf-8")`.
2. For CSV/JSON exports use UTF-8 by default.
3. When producing publication assets, keep decimal/locale formatting in content, not in file encoding.

## Legacy compatibility

Some historical artifacts under `outputs/` may contain non-UTF8 content. They are treated as legacy and must not be used as default inputs in production scripts.
