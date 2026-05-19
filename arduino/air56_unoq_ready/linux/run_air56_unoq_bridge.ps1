param(
  [string]$SerialPort = "COM5",
  [int]$Baud = 921600,
  [string]$Mode = "hybrid"
)

$Root = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Root
$Root = Split-Path -Parent $Root
if ($env:MIC_THEORY_ROOT) {
  $Root = $env:MIC_THEORY_ROOT
}
if ($env:SERIAL_PORT -and $PSBoundParameters.ContainsKey("SerialPort") -eq $false) {
  $SerialPort = $env:SERIAL_PORT
}
if ($env:BAUD -and $PSBoundParameters.ContainsKey("Baud") -eq $false) {
  $Baud = [int]$env:BAUD
}
if ($env:MODE -and $PSBoundParameters.ContainsKey("Mode") -eq $false) {
  $Mode = $env:MODE
}
$ConfigPath = if ($env:CONFIG) { $env:CONFIG } elseif ($env:CONFIG_PATH) { $env:CONFIG_PATH } else { "$Root\config\env_research_air56_025kw.py" }
$ExtraArgs = @()
if (($env:CRC -eq $null) -or ($env:CRC -eq "1")) {
  $ExtraArgs += "--crc"
}
if (($env:DISABLE_ON_FAULT -eq $null) -or ($env:DISABLE_ON_FAULT -eq "1")) {
  $ExtraArgs += "--disable-on-fault"
}

python "$Root\tools\air56_unoq_bridge.py" `
  --transport serial `
  --serial-port $SerialPort `
  --baud $Baud `
  --config $ConfigPath `
  --mode $Mode `
  @ExtraArgs
