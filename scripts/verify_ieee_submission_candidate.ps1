Param(
  [string]$Step28Dir = "paper/ieee_2026/data/step28/20260303_ai_config_locked_nodrift",
  [string]$IeeeRoot = "paper/ieee_2026",
  [string]$GuardrailsPolicy = "paper/ieee_2026/guardrails_policy.json",
  [string]$Manuscript = "paper/ieee_2026/manuscript.md",
  [string]$Tag = "",
  [switch]$AllowDirty = $true,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsRest
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$cmd = @(
  "tools/verify_ieee_submission_candidate.py",
  "--step28-dir", $Step28Dir,
  "--ieee-root", $IeeeRoot,
  "--guardrails-policy", $GuardrailsPolicy,
  "--manuscript", $Manuscript,
  "--strict"
)
if (-not [string]::IsNullOrWhiteSpace($Tag)) {
  $cmd += @("--tag", $Tag)
}
if ($AllowDirty) {
  $cmd += "--allow-dirty"
}
if ($ArgsRest) {
  $cmd += $ArgsRest
}
python @cmd
