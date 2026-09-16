param([ValidateSet("win-x64","linux-x64")][string]$Runtime="win-x64", [ValidateSet("full","cpu","cuda-windows","cuda-linux")][string]$Backend="full", [switch]$FullInstallation)
$ErrorActionPreference="Stop"
& (Join-Path $PSScriptRoot "deliver.ps1") -Target $Runtime -CpuOnly:($Backend -eq "cpu") -Full:$FullInstallation
