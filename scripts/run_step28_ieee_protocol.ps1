Param(
  [string]$OutRoot = "outputs/progress_step28_ieee_v1",
  [string]$Seeds = "101,202,303,404,505",
  [string]$Motors = "air56,al31,ao2",
  [string]$Scenarios = "speed_step,ramp,load_step,start_stop",
  [double]$SeedPerturbLevel = 0.2,
  [ValidateSet("ai", "rule")] [string]$MicMode = "rule",
  [bool]$SkipAir56Tune = $true
)

$ErrorActionPreference = "Stop"

$modeA = Join-Path $OutRoot "mode1_foc_encoder_vs_mic_sensorless"
$modeB = Join-Path $OutRoot "mode2_foc_sensorless_vs_mic_sensorless"

$argsA = @(
  "tools/step27_pipeline.py",
  "--motors", $Motors,
  "--seeds", $Seeds,
  "--scenarios", $Scenarios,
  "--out-dir", $modeA,
  "--foc-feedback-mode", "encoder",
  "--mic-feedback-mode", "sensorless",
  "--mic-mode", $MicMode,
  "--seed-perturbation",
  "--seed-perturb-level", "$SeedPerturbLevel"
)
if ($SkipAir56Tune) { $argsA += @("--skip-air56-tune") }

$argsB = @(
  "tools/step27_pipeline.py",
  "--motors", $Motors,
  "--seeds", $Seeds,
  "--scenarios", $Scenarios,
  "--out-dir", $modeB,
  "--foc-feedback-mode", "sensorless",
  "--mic-feedback-mode", "sensorless",
  "--mic-mode", $MicMode,
  "--seed-perturbation",
  "--seed-perturb-level", "$SeedPerturbLevel"
)
if ($SkipAir56Tune) { $argsB += @("--skip-air56-tune") }

Write-Host "[step28] mode1 run: python $($argsA -join ' ')"
python @argsA
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[step28] mode2 run: python $($argsB -join ' ')"
python @argsB
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[step28] building ieee summary..."
python tools/build_step28_ieee_summary.py --mode1-dir $modeA --mode2-dir $modeB --out-dir $OutRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[step28] done"
Write-Host "[step28] summary: $OutRoot/step28_ieee_summary.md"
