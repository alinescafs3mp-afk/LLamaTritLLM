$ErrorActionPreference = 'Stop'
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw 'WinGet is missing. Install Microsoft App Installer, or install .NET 10 SDK, Git, and VC++ x64 Redistributable from their official installers.'
}
foreach ($id in @('Microsoft.DotNet.SDK.10', 'Git.Git', 'Microsoft.VCRedist.2015+.x64')) {
    & winget install --exact --id $id --accept-package-agreements --accept-source-agreements --silent
    # WinGet may return a nonzero code when an up-to-date package is already installed.
    if ($LASTEXITCODE -ne 0) { Write-Warning "WinGet returned $LASTEXITCODE for $id. Verify that this package is installed." }
}
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
& dotnet --info
if ($LASTEXITCODE -ne 0) { throw 'The .NET SDK is not available. Restart the terminal after installing it.' }
Write-Host 'Ready for CPU build. NVIDIA drivers were not changed.'
