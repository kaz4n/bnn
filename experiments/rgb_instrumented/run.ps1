param(
    [string]$Python = 'C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe',
    [string]$Vivado = 'C:\Xilinx\Vivado\2016.4\bin\vivado.bat',
    [switch]$SkipBuild,
    [string]$RunName = ('physical_' + [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss'))
)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$repositoryRoot = Split-Path (Split-Path $experimentRoot -Parent) -Parent
$buildRoot = Join-Path $experimentRoot 'build'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must be one plain directory name.' }
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
if (-not $SkipBuild) {
    Push-Location $buildRoot
    try {
        & $Vivado -mode batch -source (Join-Path $experimentRoot 'hardware/build.tcl') -nojournal -nolog
        if ($LASTEXITCODE -ne 0) { throw 'Vivado failed; acquisition was not started.' }
    } finally { Pop-Location }
}
Push-Location $repositoryRoot
try {
    $runDirectory = Join-Path (Join-Path $experimentRoot 'results') $RunName
    # This command deliberately programs the volatile FPGA design.
    & $Python -m experiments.rgb_instrumented.capture --source fpga --run-dir $runDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Physical acquisition failed; no model fallback is allowed.' }
    & $Python -m experiments.rgb_instrumented.evaluate --run-dir $runDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation recorded failures; inspect metrics.json.' }
    Write-Output (Join-Path $runDirectory 'results.md')
} finally { Pop-Location }
