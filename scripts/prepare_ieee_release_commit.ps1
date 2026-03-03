Param(
  [string]$Step28Dir = "paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift",
  [string]$IeeeRoot = "paper/ieee_2026",
  [string]$Tag = "",
  [switch]$Apply = $false,
  [switch]$Push = $false,
  [switch]$AllowDirty = $true,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsRest
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$cmd = @(
  "tools/prepare_ieee_release_commit.py",
  "--step28-dir", $Step28Dir,
  "--ieee-root", $IeeeRoot
)
if (-not [string]::IsNullOrWhiteSpace($Tag)) {
  $cmd += @("--tag", $Tag)
}
if ($Apply) {
  $cmd += "--apply"
}
if ($Push) {
  $cmd += "--push"
}
if ($AllowDirty) {
  $cmd += "--allow-dirty"
}
if ($ArgsRest) {
  $cmd += $ArgsRest
}
python @cmd
