# Project Structure (Refactor 2026-03-03)

This document fixes the project layout contract and reduces ad-hoc script drift.

## Top-level responsibilities
- `mic_ai/`: core runtime, models, metrics, training and analysis primitives.
- `config/`: motor/environment configs and fixed evaluation defaults.
- `tools/`: research/benchmark pipelines and publication data builders.
- `scripts/`: orchestration wrappers (`.ps1`/`.sh`) around tool entrypoints.
- `tests/`: unit/smoke/regression checks.
- `paper/`: publication-ready data/figures/tables.
- `outputs/`: generated artifacts (not source of truth for code).

## New shared utility layer
To avoid duplicated logic in pipelines, common helpers are centralized:

- `tools/common_utils.py`
  - CSV/int parsing helpers.
  - CSV/JSON read-write helpers.
  - basic `mean/std`.

- `tools/checkpoint_registry.py`
  - checkpoint registry loading (`motors`, `configs`, `by_motor` formats).
  - deterministic checkpoint candidate resolution.
  - configurable priority: registry-first or env-first.

## Pipeline conventions
- Step27/Step28 scripts must use shared helpers from `tools/common_utils.py`.
- Any checkpoint resolution in tooling must use `tools/checkpoint_registry.py`.
- CLI scripts should keep backward-compatible output artifact names.

## Backward compatibility note
- `tools/step27_pipeline.py` still exposes legacy helper names
  (`_resolve_checkpoint`, `_load_checkpoint_registry`, etc.) so existing tests
  and scripts continue to work, but implementation is now delegated to shared modules.
