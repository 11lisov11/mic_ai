#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT="${MIC_THEORY_ROOT:-$ROOT}"
SERIAL_PORT="${1:-${SERIAL_PORT:-/dev/ttyHS0}}"
BAUD="${BAUD:-921600}"
CONFIG="${CONFIG:-${CONFIG_PATH:-$ROOT/config/env_research_air56_025kw.py}}"
MODE="${MODE:-hybrid}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ "${CRC:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--crc)
fi
if [[ "${DISABLE_ON_FAULT:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-on-fault)
fi

"$PYTHON_BIN" "$ROOT/tools/air56_unoq_bridge.py" \
  --transport serial \
  --serial-port "$SERIAL_PORT" \
  --baud "$BAUD" \
  --config "$CONFIG" \
  --mode "$MODE" \
  "${EXTRA_ARGS[@]}"
