# Install / uninstall the Mellonella Windows APO.
#
# Usage (Administrator PowerShell):
#   .\install-apo.ps1            # register the DLL
#   .\install-apo.ps1 -Uninstall # unregister and clean up
#
# Run `cargo build -p mellonella-apo --release` first (native MSVC
# build), or copy the artefact from a cross-compiled
# `target\x86_64-pc-windows-gnu\release\mellonella_apo.dll`.
#
# This script handles the regsvr32 dance only — attaching the APO
# CLSID to a specific capture endpoint's PKEY_FX_StreamEffectClsidList
# is a manual step described in docs/apo.md, because the right
# endpoint depends on which microphone the user wants to filter.

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [string]$DllPath = "$PSScriptRoot\..\rust\target\release\mellonella_apo.dll",
    [string]$InstallDir = "$Env:ProgramFiles\Mellonella"
)

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]'Administrator')) {
    throw "install-apo.ps1 must run from an elevated (Administrator) PowerShell."
}

if ($Uninstall) {
    $target = Join-Path $InstallDir 'mellonella_apo.dll'
    if (Test-Path $target) {
        Write-Host "unregistering $target"
        Start-Process -FilePath regsvr32.exe -ArgumentList "/u /s `"$target`"" -Wait
        Remove-Item $target -Force
    }
    Write-Host "restarting Audiosrv"
    Restart-Service Audiosrv -Force
    Write-Host "done. Remember to remove the Mellonella CLSID from your microphone's PKEY_FX_StreamEffectClsidList registry value (see docs/apo.md)."
    return
}

if (-not (Test-Path $DllPath)) {
    throw "DLL not found at $DllPath. Build first: cargo build -p mellonella-apo --release"
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$target = Join-Path $InstallDir 'mellonella_apo.dll'
Copy-Item $DllPath $target -Force
Write-Host "copied → $target"

Write-Host "registering COM class"
Start-Process -FilePath regsvr32.exe -ArgumentList "/s `"$target`"" -Wait

Write-Host "restarting Audiosrv"
Restart-Service Audiosrv -Force

Write-Host @"

mellonella_apo.dll is registered.

Next: attach it to a capture endpoint's effect chain.
See docs/apo.md for the registry walk-through.

"@
