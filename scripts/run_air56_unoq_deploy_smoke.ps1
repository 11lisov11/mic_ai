param(
  [string]$OutJson = ""
)

$ErrorActionPreference = "Stop"
$ArgsList = @("tools/run_air56_unoq_deploy_smoke.py")
if ($OutJson) {
  $ArgsList += @("--out-json", $OutJson)
}
python @ArgsList
