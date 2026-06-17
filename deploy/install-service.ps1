<#
Install LLMConfig as an always-on Windows service on .40 + open the firewall port.

Run elevated (the service needs admin rights to Start/Restart the Ollama service):
    powershell -ExecutionPolicy Bypass -File deploy\install-service.ps1

Prefers NSSM if present; otherwise registers a Scheduled Task at logon (RunLevel
Highest). Assumes a venv at <repo>\.venv with `pip install -e .` already done.
#>
[CmdletBinding()]
param(
    [string]$RepoPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [int]$Port = 11430,
    [string]$ServiceName = "LLMConfig"
)
$ErrorActionPreference = "Stop"

$py = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    throw "venv python not found at $py. Create it first: python -m venv .venv; .venv\Scripts\pip install -e ."
}

# --- Firewall: allow the control port on the LAN (mirrors Ollama's rule) ---
if (-not (Get-NetFirewallRule -DisplayName "LLMConfig $Port" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "LLMConfig $Port" -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
    Write-Host "Added Windows Firewall inbound rule for TCP $Port."
}

$uvArgs = "-m uvicorn llmconfig.main:app --host 0.0.0.0 --port $Port"
$nssm = (Get-Command nssm.exe -ErrorAction SilentlyContinue).Source

if ($nssm) {
    & $nssm install $ServiceName $py $uvArgs
    & $nssm set $ServiceName AppDirectory $RepoPath
    & $nssm set $ServiceName Start SERVICE_AUTO_START
    & $nssm set $ServiceName DisplayName "LLMConfig (Ollama/vLLM GPU arbiter)"
    & $nssm start $ServiceName
    Write-Host "Installed and started service '$ServiceName' via NSSM (runs as LocalSystem = elevated)."
}
else {
    $action  = New-ScheduledTaskAction -Execute $py -Argument $uvArgs -WorkingDirectory $RepoPath
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $set     = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $ServiceName -Action $action -Trigger $trigger `
        -RunLevel Highest -Settings $set -Force | Out-Null
    Write-Host "NSSM not found. Registered Scheduled Task '$ServiceName' (at logon, elevated)."
    Write-Host "Start it now with:  Start-ScheduledTask -TaskName $ServiceName"
}
Write-Host "UI: http://192.168.1.40:$Port/"
