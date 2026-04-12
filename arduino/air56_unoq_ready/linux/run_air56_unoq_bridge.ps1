param(
  [string]$SerialPort = "COM5",
  [int]$Baud = 921600
)

$Root = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Root
$Root = Split-Path -Parent $Root

python "$Root\tools\air56_unoq_bridge.py" `
  --transport serial `
  --serial-port $SerialPort `
  --baud $Baud `
  --config "$Root\config\env_research_air56_025kw.py" `
  --mode hybrid `
  --crc `
  --disable-on-fault
