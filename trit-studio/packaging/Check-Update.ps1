param([string]$Root = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar)
function Resolve-Safe([string]$relative) {
    if ([string]::IsNullOrWhiteSpace($relative) -or $relative.Contains('\') -or $relative.Contains(':') -or ($relative.Split('/') | Where-Object { $_ -eq '..' -or $_ -eq '.' -or $_ -eq '' })) { throw "Unsafe manifest path: $relative" }
    $path = [IO.Path]::GetFullPath((Join-Path $Root $relative))
    if (-not $path.StartsWith($Root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside installation: $relative" }
    $part = $path
    while ($part -and $part.Length -ge $Root.Length) {
        if ((Test-Path -LiteralPath $part) -and ((Get-Item -LiteralPath $part -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Linked installation path: $relative" }
        $part = [IO.Path]::GetDirectoryName($part)
    }
    return $path
}
function Verify-Files($entries, [bool]$runtime) {
    if ($null -eq $entries -or @($entries.PSObject.Properties).Count -eq 0) { throw 'Empty integrity manifest cannot certify an installation.' }
    foreach ($entry in $entries.PSObject.Properties) {
        $path = Resolve-Safe $entry.Name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $($entry.Name). Merge into the EXISTING installation; an update is not standalone." }
        $expected = if ($runtime) { $entry.Value.sha256 } else { $entry.Value }
        if ($expected -notmatch '^[a-fA-F0-9]{64}$') { throw "Invalid digest: $($entry.Name)" }
        if ($runtime -and (Get-Item -LiteralPath $path).Length -ne $entry.Value.bytes) { throw "Runtime size mismatch: $($entry.Name). Do not start this mixed installation." }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash -ne $expected) { throw "File mismatch: $($entry.Name). Restore a coherent installation; do not download or delete random CUDA DLLs." }
    }
}
try {
    if (Test-Path -LiteralPath (Join-Path $Root 'UPDATE_SHA256SUMS.json')) {
        Write-Host '[update] Checking owned files...'
        Verify-Files (Get-Content -Raw -LiteralPath (Join-Path $Root 'UPDATE_SHA256SUMS.json') | ConvertFrom-Json) $false
        $runtime = Get-Content -Raw -LiteralPath (Join-Path $Root 'REQUIRED_RUNTIME_FILES.json') | ConvertFrom-Json
        if ($runtime.schema -ne 1 -or $runtime.target -ne 'win-x64') { throw 'Unsupported runtime manifest.' }
        Write-Host '[runtime] Checking retained vendor files (may take time; no download)...'
        Verify-Files $runtime.files $true
    } else {
        Write-Host '[full] Checking portable files...'
        Verify-Files (Get-Content -Raw -LiteralPath (Join-Path $Root 'SHA256SUMS.json') | ConvertFrom-Json) $false
    }
    Write-Host 'Integrity OK. This does not replace actual CPU/CUDA and UI tests.'
    exit 0
} catch { Write-Host ('FAILED: ' + $_.Exception.Message) -ForegroundColor Red; exit 1 }
