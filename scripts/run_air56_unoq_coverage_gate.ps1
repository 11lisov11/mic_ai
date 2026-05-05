param(
  [string]$CoverageJson = ".tmp_pytest/coverage_air56_unoq_gate.json"
)

$ErrorActionPreference = "Stop"
python tools/check_air56_unoq_coverage_gate.py --coverage-json $CoverageJson
