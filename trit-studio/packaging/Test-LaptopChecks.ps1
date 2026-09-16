# Test the ACTUAL Run-Check implementation without running a model or altering the installation.
$ErrorActionPreference = 'Stop'
$temp = Join-Path ([IO.Path]::GetTempPath()) ('trit-check-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $tokens=$null; $parseErrors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'Check-laptop.ps1'),[ref]$tokens,[ref]$parseErrors)
    if($parseErrors.Count){ throw 'Check-laptop.ps1 does not parse.' }
    $functions=@($ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Run-Check'},$true))
    if($functions.Count -ne 1){ throw 'Expected one real Run-Check function.' }
    . ([scriptblock]::Create($functions[0].Extent.Text))
    $reportDirectory=$temp
    $checks=New-Object 'System.Collections.Generic.List[object]'
    $exe=Join-Path $PSHOME 'powershell.exe'
    if(-not(Test-Path -LiteralPath $exe)){ $exe=Join-Path $PSHOME 'pwsh.exe' }
    $fixture=Join-Path $temp 'child fixture.ps1'
    [IO.File]::WriteAllText($fixture,'param([int]$Code) [Console]::Error.WriteLine("expected native diagnostic"); [Console]::Out.WriteLine("expected stdout"); exit $Code')
    Run-Check 'stderr-success' $exe @('-NoProfile','-ExecutionPolicy','Bypass','-File',$fixture,'-Code','0')
    if($checks.Count -ne 1 -or $checks[0].exitCode -ne 0){throw 'stderr with exit0 was not a successful check.'}
    if(-not((Get-Content -Raw -LiteralPath (Join-Path $temp 'stderr-success.log')) -match 'expected native diagnostic')){throw 'stderr was discarded.'}
    $failed=$false
    try { Run-Check 'stderr-failure' $exe @('-NoProfile','-ExecutionPolicy','Bypass','-File',$fixture,'-Code','7') } catch { $failed=$true }
    if(-not $failed -or $checks[1].exitCode -ne 7){throw 'Nonzero native exit was hidden.'}
    if($ErrorActionPreference -ne 'Stop'){throw 'Run-Check leaked its temporary error preference.'}
    Write-Host 'PASS laptop runner: captured stderr, real exit0/nonzero and restored preference.'
    exit 0
} catch { Write-Host ('FAILED laptop runner test: '+$_.Exception.Message); exit 1 }
finally { Remove-Item -LiteralPath $temp -Recurse -Force }
