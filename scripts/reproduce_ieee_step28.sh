#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi

"${PY_BIN}" tools/reproduce_ieee_step28.py "$@"
