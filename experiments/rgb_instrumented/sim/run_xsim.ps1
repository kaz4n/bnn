param(
    [string]$VivadoBin = 'C:/Xilinx/Vivado/2016.4/bin'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$experimentDirectory = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $PSScriptRoot 'build'
$rtlFile = Join-Path $experimentDirectory 'rtl/rgb_bayer_capture.sv'
$testbenchFile = Join-Path $PSScriptRoot 'tb_rgb_bayer_capture.sv'

foreach ($toolName in @('xvlog.bat', 'xelab.bat', 'xsim.bat')) {
    if (-not (Test-Path -LiteralPath (Join-Path $VivadoBin $toolName))) {
        throw "Missing $toolName in $VivadoBin. Pass -VivadoBin with an installed Vivado bin directory."
    }
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
Push-Location -LiteralPath $buildDirectory
try {
    & (Join-Path $VivadoBin 'xvlog.bat') --sv $rtlFile $testbenchFile
    if ($LASTEXITCODE -ne 0) { throw "RTL simulation compilation failed: $LASTEXITCODE" }
    & (Join-Path $VivadoBin 'xelab.bat') tb_rgb_bayer_capture -s rgb_capture_tb -debug typical
    if ($LASTEXITCODE -ne 0) { throw "RTL simulation elaboration failed: $LASTEXITCODE" }
    & (Join-Path $VivadoBin 'xsim.bat') rgb_capture_tb -runall
    if ($LASTEXITCODE -ne 0) { throw "RTL simulation execution failed: $LASTEXITCODE" }
    $simulationLog = Get-Content -LiteralPath (Join-Path $buildDirectory 'xsim.log') -Raw
    if ($simulationLog -notmatch 'RTL simulation PASS: all tests, 21504 frame bytes verified\.') {
        throw 'RTL simulation did not produce its complete self-check PASS marker.'
    }
    Write-Output 'RTL simulation verified: 32x32, 64x64, 128x128; 21504 bytes; protocol and reset cases.'
}
finally {
    Pop-Location
}
