param([switch]$CpuOnly)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($PSScriptRoot)
$reportDirectory = Join-Path $root ("diagnostics/" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$checks = New-Object 'System.Collections.Generic.List[object]'
function Run-Check([string]$Name, [string]$Exe, [string[]]$Arguments) {
    Write-Host "`n[$Name] Running..."
    $log = Join-Path $reportDirectory ($Name + ".log")
    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "Missing executable: $Exe" }
    # Windows PowerShell 5.1 may turn redirected native stderr into NativeCommandError.
    # A diagnostic line is not an exit-code failure; keep it in the log, then check the actual exit.
    # Reset the value first: a failure to start must not reuse a previous process's success code.
    $savedPreference = $ErrorActionPreference
    $global:LASTEXITCODE = $null
    try {
        $ErrorActionPreference = 'Continue'
        & $Exe @Arguments 2>&1 | Tee-Object -FilePath $log -ErrorAction Stop | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($null -eq $code) { throw "$Name did not produce a process exit code. Read $log" }
    $checks.Add([pscustomobject]@{name=$Name; exitCode=$code; log=$log})
    if ($code -ne 0) { throw "$Name failed with exit code $code. Read $log" }
}
try {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "Check-Update.ps1") -Root $root
    if ($LASTEXITCODE -ne 0) { throw "Installation integrity/runtime compatibility check failed. No runtime was downloaded or replaced." }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "Test-LaptopChecks.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Laptop-check runner regression failed." }
    $checks.Add([pscustomobject]@{name="integrity";exitCode=0})
    Run-Check "core-tests" (Join-Path $root "checks/TritStudio.Tests.exe") @()
    Run-Check "cpu-doctor" (Join-Path $root "trainer/TritStudio.Trainer.exe") @("--doctor")
    Run-Check "cpu-training" (Join-Path $root "trainer/TritStudio.Trainer.exe") @("--self-test")
    Run-Check "cpu-benchmark" (Join-Path $root "trainer/TritStudio.Trainer.exe") @("--benchmark","--output",(Join-Path $reportDirectory "cpu-benchmark.json"))
    if (-not $CpuOnly) {
        if (-not (Test-Path -LiteralPath (Join-Path $root "trainer-cuda/TritStudio.Trainer.exe"))) { throw "CUDA trainer is absent. This is a CPU-only package. Use -CpuOnly only for explicit CPU-only checks." }
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi 2>&1 | Out-File -FilePath (Join-Path $reportDirectory "nvidia-smi.log") }
        Run-Check "cuda-doctor" (Join-Path $root "trainer-cuda/TritStudio.Trainer.exe") @("--doctor")
        Run-Check "cuda-training" (Join-Path $root "trainer-cuda/TritStudio.Trainer.exe") @("--self-test","--cuda")
        Run-Check "cuda-benchmark" (Join-Path $root "trainer-cuda/TritStudio.Trainer.exe") @("--benchmark","--cuda","--output",(Join-Path $reportDirectory "cuda-benchmark.json"))
    }
    $result = [pscustomobject]@{status="passed"; cpuOnly=[bool]$CpuOnly; checks=$checks; manualDesktopCheck="PENDING"; timestamp=(Get-Date -Format o)}
    $result | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $reportDirectory "RESULT.json")
    Write-Host "`nPASS. Reports: $reportDirectory. Desktop clicks still require the manual checklist."
    exit 0
} catch {
    $result = [pscustomobject]@{status="failed"; error=$_.Exception.Message; checks=$checks; timestamp=(Get-Date -Format o)}
    $result | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $reportDirectory "RESULT.json")
    Write-Host "`nFAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Reports: $reportDirectory"
    exit 1
}
