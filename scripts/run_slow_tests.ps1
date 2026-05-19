param(
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
python -m pytest -q -m "slow and not hardware" @ExtraArgs
