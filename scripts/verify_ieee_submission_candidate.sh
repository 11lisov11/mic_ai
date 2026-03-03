#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

step28_dir="${1:-paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift}"
ieee_root="${2:-paper/ieee_2026}"
guardrails_policy="${3:-paper/ieee_2026/guardrails_policy.json}"
manuscript="${4:-paper/ieee_2026/manuscript.md}"
tag="${5:-}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

cmd=(
  "${PY_BIN}" tools/verify_ieee_submission_candidate.py
  --step28-dir "${step28_dir}"
  --ieee-root "${ieee_root}"
  --guardrails-policy "${guardrails_policy}"
  --manuscript "${manuscript}"
  --strict
  --allow-dirty
)
if [[ -n "${tag}" ]]; then
  cmd+=(--tag "${tag}")
fi

"${cmd[@]}" "$@"
