<#
Install LLMConfig as an always-on task/service on .40 + open the firewall port.

Run elevated (it needs admin to add the firewall rule and to register a task /
service that can Start/Restart the Ollama service):
    powershell -ExecutionPolicy Bypass -File deploy\install-service.ps1

-Method Task (default): Scheduled Task at logon, RunLevel Highest, running as the
    interactive user. Required for the vLLM path -- the app shells into WSL
    (`wsl.exe -u <user> ...`) for serve.sh + the keepalive, and WSL needs the
    user's session, which a LocalSystem service (Session 0) does not have.
-Method Nssm: true Windows service via NSSM (auto-start at boot, runs as
    LocalSystem). Ollama control works, but WSL/vLLM control will very likely
    fail from Session 0 -- use only for an Ollama-only deployment.

Assumes a venv at <repo>\.venv with `pip install -e .` already done.
#>
[CmdletBinding()]
param(
    [string]$RepoPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [int]$Port = 11430,
    [string]$ServiceName = "LLMConfig",
    [ValidateSet("Task", "Nssm")]
    [string]$Method = "Task"
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

if ($Method -eq "Nssm") {
    $nssm = (Get-Command nssm.exe -ErrorAction SilentlyContinue).Source
    if (-not $nssm) { throw "NSSM not found on PATH. Install it, or use -Method Task." }
    & $nssm install $ServiceName $py $uvArgs
    & $nssm set $ServiceName AppDirectory $RepoPath
    & $nssm set $ServiceName Start SERVICE_AUTO_START
    & $nssm set $ServiceName DisplayName "LLMConfig (Ollama/vLLM GPU arbiter)"
    & $nssm start $ServiceName
    Write-Host "Installed and started service '$ServiceName' via NSSM (runs as LocalSystem = elevated)."
    Write-Host "WARNING: WSL/vLLM control may fail from Session 0; this is best for Ollama-only."
}
else {
    # Run as the invoking interactive user so WSL works; RunLevel Highest gives the
    # elevation needed to Start/Restart the Ollama service.
    $userId  = "$env:USERDOMAIN\$env:USERNAME"
    $action  = New-ScheduledTaskAction -Execute $py -Argument $uvArgs -WorkingDirectory $RepoPath
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
    $set     = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
    $principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Highest
    Register-ScheduledTask -TaskName $ServiceName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $set -Force | Out-Null
    Start-ScheduledTask -TaskName $ServiceName
    Write-Host "Registered + started Scheduled Task '$ServiceName' (at logon as $userId, elevated)."
}
Write-Host "UI: http://192.168.1.40:$Port/"
