#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/tools/delta_ms300_modbus_bridge.py" --dry-run self-check
python "$ROOT/tools/delta_ms300_modbus_bridge.py" --dry-run read-once
python "$ROOT/tools/delta_ms300_modbus_bridge.py" --dry-run stage0 --probe-frequency-hz 1.0
python -m pytest -q tests/test_delta_ms300_modbus.py
