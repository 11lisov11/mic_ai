#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT="${MIC_THEORY_ROOT:-$ROOT}"
SERIAL_PORT="${1:-${SERIAL_PORT:-/dev/ttyHS0}}"
BAUD="${BAUD:-921600}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/config/env_research_air56_025kw.py}"
MODE="${MODE:-hybrid}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/tools/air56_unoq_bridge.py" \
  --transport serial \
  --serial-port "$SERIAL_PORT" \
  --baud "$BAUD" \
  --config "$CONFIG_PATH" \
  --mode "$MODE" \
  --crc \
  --disable-on-fault
