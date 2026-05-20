#!/usr/bin/env bash
set -euo pipefail
ROOT="${MIC_THEORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONFIG="${CONFIG:-$ROOT/config/vfd_delta_ms300_air56.json}"
python "$ROOT/tools/delta_ms300_modbus_bridge.py" --config "$CONFIG" "$@"
