param(
    [string]$Root = (Resolve-Path "$PSScriptRoot\..\..\..").Path,
    [string]$Config = "",
    [switch]$DryRun,
    [switch]$AllowWrite,
    [double]$ProbeHz = 1.0
)

if (-not $Config) {
    $Config = Join-Path $Root "config\vfd_delta_ms300_air56.json"
}

$argsList = @("tools\delta_ms300_modbus_bridge.py", "--config", $Config)
if ($DryRun) { $argsList += "--dry-run" }
if ($AllowWrite) { $argsList += "--allow-write" }
$argsList += @("stage0", "--probe-frequency-hz", [string]$ProbeHz)
if ($AllowWrite) { $argsList += "--write-probe" }

Push-Location $Root
try {
    python @argsList
} finally {
    Pop-Location
}
