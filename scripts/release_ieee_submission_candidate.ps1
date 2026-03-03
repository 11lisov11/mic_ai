Param(
  [string]$OutRoot = "outputs/release_ieee_submission_candidate",
  [string]$Tag = "",
  [string]$MicMode = "rule",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsRest
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$tagArg = @()
if ([string]::IsNullOrWhiteSpace($Tag)) {
  $Tag = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
}
$tagArg = @("--package-tag", $Tag)

python tools/reproduce_ieee_step28.py `
  --out-root $OutRoot `
  --mic-mode $MicMode `
  --promote-release `
  --strict-verify `
  --freeze-require-publication-assets `
  --freeze-require-release-assets `
  --guardrails-policy "paper/ieee_2026/guardrails_policy.json" `
  @tagArg `
  @ArgsRest
