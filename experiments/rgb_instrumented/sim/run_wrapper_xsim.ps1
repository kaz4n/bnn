param(
    [string]$VivadoBin = 'C:/Xilinx/Vivado/2016.4/bin',
    [string]$Frontend = 'C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl/cw305_usb_reg_fe.v'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$experimentDirectory = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $PSScriptRoot 'build/wrapper'
$rtlFile = Join-Path $experimentDirectory 'rtl/rgb_bayer_capture.sv'
$wrapperFile = Join-Path $experimentDirectory 'hardware/cw305_rgb_top.sv'
$testbenchFile = Join-Path $PSScriptRoot 'tb_cw305_rgb_top.sv'
$clockStubFile = Join-Path $PSScriptRoot 'bufg_sim_stub.sv'
$waveformScript = (Join-Path $PSScriptRoot 'wrapper_waveforms.tcl').Replace('\', '/')

foreach ($toolName in @('xvlog.bat', 'xelab.bat', 'xsim.bat')) {
    if (-not (Test-Path -LiteralPath (Join-Path $VivadoBin $toolName))) {
        throw "Missing $toolName in $VivadoBin. Pass -VivadoBin with an installed Vivado bin directory."
    }
}
if (-not (Test-Path -LiteralPath $Frontend)) {
    throw 'Missing stock cw305_usb_reg_fe.v. Pass -Frontend with its installed path.'
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
Push-Location -LiteralPath $buildDirectory
try {
    & (Join-Path $VivadoBin 'xvlog.bat') --sv $rtlFile $wrapperFile $Frontend $clockStubFile $testbenchFile
    if ($LASTEXITCODE -ne 0) { throw "Wrapper RTL simulation compilation failed: $LASTEXITCODE" }
    & (Join-Path $VivadoBin 'xelab.bat') tb_cw305_rgb_top -s rgb_wrapper_tb -debug typical
    if ($LASTEXITCODE -ne 0) { throw "Wrapper RTL simulation elaboration failed: $LASTEXITCODE" }
    & (Join-Path $VivadoBin 'xsim.bat') rgb_wrapper_tb -tclbatch $waveformScript -onfinish stop -onerror quit
    if ($LASTEXITCODE -ne 0) { throw "Wrapper RTL simulation execution failed: $LASTEXITCODE" }
    $simulationLog = Get-Content -LiteralPath (Join-Path $buildDirectory 'xsim.log') -Raw
    if ($simulationLog -notmatch 'Wrapper RTL simulation PASS: all tests, 17408 frame bytes verified\.') {
        throw 'Wrapper RTL simulation did not produce its complete self-check PASS marker.'
    }
    if ($simulationLog -notmatch 'Wrapper RTL simulation PASS: minimum read profile,') {
        throw 'Wrapper RTL simulation did not pass the minimum four-cycle read profile.'
    }
    Write-Output 'Wrapper RTL simulation verified: registers, held writes, chunked transfers, 32x32 and 128x128 readback with setup=1/pulse=3/cycle=4 reads.'
}
finally {
    Pop-Location
}
