<#
Register the weekly Ollama-update task on .40.

Creates a Scheduled Task "LLMConfig-OllamaUpdate" that runs deploy\update-ollama.ps1
weekly (default Sunday 04:00 -- the lab's weekly maintenance window). The updater stops
LLMConfig + both Ollama services, runs OllamaSetup silently, re-suppresses the tray, and
verifies the CUDA runner survived. It is deliberately a LOCAL Windows task (not Rundeck):
the update needs the interactive user's Windows service control + HKCU + per-user install.

Mirrors the principal/elevation used by install-service.ps1's LLMConfig task -- the same
interactive user at RunLevel Highest, needed to Stop/Start services, stop the LLMConfig
task, and reach the per-user install + HKCU.

Run elevated:
    powershell -ExecutionPolicy Bypass -File deploy\install-ollama-update.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$DayOfWeek = "Sunday",
    [string]$At = "4:00AM",
    [string]$TaskName = "LLMConfig-OllamaUpdate"
)
$ErrorActionPreference = "Stop"

$script = Join-Path $RepoPath "deploy\update-ollama.ps1"
if (-not (Test-Path $script)) { throw "update-ollama.ps1 not found at $script." }

# Run as the invoking interactive user, RunLevel Highest -- same as the LLMConfig task,
# so it can Stop/Start the Ollama services + the LLMConfig task and touch the per-user install.
$userId    = "$env:USERDOMAIN\$env:USERNAME"
# Pass -RepoPath explicitly: under Task Scheduler -File, update-ollama.ps1's $PSScriptRoot
# came back empty (logs then landed in C:\logs), so don't rely on it resolving the repo.
$action    = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -RepoPath `"$RepoPath`"" -WorkingDirectory $RepoPath
$trigger   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayOfWeek -At $At
$set       = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $set -Force | Out-Null
Write-Host "Registered Scheduled Task '$TaskName' ($DayOfWeek $At, as $userId, elevated)."
Write-Host "Runs: $script"
Write-Host "Fire once now to test:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Log:                    $RepoPath\logs\ollama-update.log"
