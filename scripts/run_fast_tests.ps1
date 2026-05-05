param(
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
python -m pytest -q -m "not slow and not hardware" @ExtraArgs
