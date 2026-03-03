#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

step28_dir="${1:-paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift}"
ieee_root="${2:-paper/ieee_2026}"
tag="${3:-}"
apply="${4:-0}"
push="${5:-0}"
allow_dirty="${6:-1}"

if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

cmd=(
  "${PY_BIN}" tools/prepare_ieee_release_commit.py
  --step28-dir "${step28_dir}"
  --ieee-root "${ieee_root}"
)
if [[ -n "${tag}" ]]; then
  cmd+=(--tag "${tag}")
fi
if [[ "${apply}" == "1" ]]; then
  cmd+=(--apply)
fi
if [[ "${push}" == "1" ]]; then
  cmd+=(--push)
fi
if [[ "${allow_dirty}" == "1" ]]; then
  cmd+=(--allow-dirty)
fi

"${cmd[@]}" "$@"
