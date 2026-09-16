param([ValidateSet("win-x64","linux-x64")][string]$Target = "win-x64", [switch]$CpuOnly, [switch]$Full)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    $arguments = @("run", "--project", "tools/Delivery/Delivery.csproj", "-c", "Release", "--", "--target", $Target)
    if ($CpuOnly) { $arguments += "--cpu-only" }
    if ($Full) { $arguments += "--full" }
    & dotnet @arguments
    if ($LASTEXITCODE -ne 0) { throw "Delivery failed with exit code $LASTEXITCODE. Read artifacts/build-logs." }
} finally { Pop-Location }
