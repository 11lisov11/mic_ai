$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
    python tools\delta_ms300_modbus_bridge.py --dry-run self-check
    python tools\delta_ms300_modbus_bridge.py --dry-run read-once
    python tools\delta_ms300_modbus_bridge.py --dry-run stage0 --probe-frequency-hz 1.0
    python -m pytest -q tests\test_delta_ms300_modbus.py
} finally {
    Pop-Location
}
