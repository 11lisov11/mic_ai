#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERIAL_PORT="${1:-/dev/ttyHS0}"
BAUD="${BAUD:-921600}"

python "$ROOT/tools/air56_unoq_bridge.py" \
  --transport serial \
  --serial-port "$SERIAL_PORT" \
  --baud "$BAUD" \
  --config "$ROOT/config/env_research_air56_025kw.py" \
  --mode hybrid \
  --crc \
  --disable-on-fault
