<#
Safely update Ollama on .40 (both NSSM instances) on a schedule, then verify the
CUDA runner survived. Replaces the manual "Stop-Service -> OllamaSetup -> Start-Service"
dance documented in install-companion.ps1.

Why this exists: Ollama's tray auto-updater is disabled (it can't stop the NSSM-managed
ollama.exe, so its in-place update hits "DeleteFile ... Access is denied", rolls back, and
wipes the CUDA runner -> silent CPU-only fallback: library=cpu, total_vram=0). The safe
sequence is: stop LLMConfig (so it can't Start-Service ollama mid-install and re-lock the
binary), stop BOTH Ollama services + the tray, run OllamaSetup silently, re-suppress the
tray the installer re-enables, restart, and then PROVE the runner offloads to GPU (service
"Running" alone is not proof -- that's the exact 2026-06-18 corruption this guards against).

Both services run the SAME per-user ollama.exe, so the binary is locked until every Ollama
process is stopped. NSSM AppEnvironmentExtra (OLLAMA_HOST, GPU pinning, OLLAMA_MODELS,
OLLAMA_CONTEXT_LENGTH) lives in the service definition and survives a binary update -- no
reconfigure needed here.

Run on the box (it is the dev machine), elevated (needs to Stop/Start services, stop the
LLMConfig task, and touch the per-user install + HKCU):
    powershell -ExecutionPolicy Bypass -File deploy\update-ollama.ps1            # skip if already latest
    powershell -ExecutionPolicy Bypass -File deploy\update-ollama.ps1 -Force     # reinstall regardless

Normally invoked weekly by the LLMConfig-OllamaUpdate scheduled task
(see deploy\install-ollama-update.ps1).
#>
[CmdletBinding()]
param(
    [string]$RepoPath = (Resolve-Path "$PSScriptRoot\..").Path,
    [string[]]$ServiceNames = @('Ollama', 'OllamaCompanion'),
    [string]$LLMConfigTask = "LLMConfig",
    [int]$Port = 11430,                 # LLMConfig's control port (freed if still held)
    [string]$OllamaExe = "",
    [string]$VerifyModel = "",          # empty => auto-pick the smallest already-pulled model
    [switch]$Force,                     # reinstall even when already on the latest version
    [string]$LogPath = ""
)
$ErrorActionPreference = "Stop"
if (-not $LogPath) { $LogPath = Join-Path $RepoPath "logs\ollama-update.log" }

# --- logging --------------------------------------------------------------
function Write-Log {
    param([string]$Message)
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    try {
        $dir = Split-Path $LogPath
        if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -Path $LogPath -Value $line
    } catch { Write-Host "  (could not write log: $($_.Exception.Message))" }
}

# --- resolve ollama.exe (same lookup as install-companion.ps1) ------------
if (-not $OllamaExe) {
    $cmd = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($cmd) { $OllamaExe = $cmd.Source }
    elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") { $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
    else { throw "ollama.exe not found. Pass -OllamaExe <path>." }
}

function Get-InstalledVersion {
    try {
        $out = & $OllamaExe --version 2>&1 | Out-String
        $m = [regex]::Match($out, '\d+\.\d+\.\d+')
        if ($m.Success) { return $m.Value }
    } catch { }
    return $null
}

function Get-LatestVersion {
    # GitHub requires a User-Agent. Any network/TLS failure => caller skips this run.
    $headers = @{ "User-Agent" = "LLMConfig-OllamaUpdater"; "Accept" = "application/vnd.github+json" }
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ollama/ollama/releases/latest" `
        -Headers $headers -TimeoutSec 30
    return ($rel.tag_name -replace '^v', '')
}

# --- existing-service helpers --------------------------------------------
function Get-PresentServices {
    $ServiceNames | Where-Object { Get-Service -Name $_ -ErrorAction SilentlyContinue }
}

# Each instance's API port, read from its NSSM OLLAMA_HOST (falls back to known defaults).
function Get-ServicePort {
    param([string]$Name)
    $nssm = (Get-Command nssm.exe -ErrorAction SilentlyContinue).Source
    if ($nssm) {
        try {
            $extra = (& $nssm get $Name AppEnvironmentExtra 2>$null) | Out-String
            $m = [regex]::Match($extra, 'OLLAMA_HOST=[^\s"]*:(\d+)')
            if ($m.Success) { return [int]$m.Groups[1].Value }
        } catch { }
    }
    if ($Name -eq 'OllamaCompanion') { return 11435 }
    return 11434
}

function Stop-OllamaProcesses {
    foreach ($name in (Get-PresentServices)) {
        Write-Log "Stopping service '$name'..."
        Stop-Service -Name $name -Force -ErrorAction SilentlyContinue
    }
    # Kill the tray + any stray ollama.exe so the per-user binary is unlocked for replace.
    Get-Process 'ollama app', 'ollama' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Start-OllamaServices {
    foreach ($name in (Get-PresentServices)) {
        Write-Log "Starting service '$name'..."
        Start-Service -Name $name -ErrorAction SilentlyContinue
    }
}

# --- tray suppression (verbatim from install-companion.ps1:51-73) ---------
# OllamaSetup re-enables the tray + login-autostart on every install, so re-run this after.
function Suppress-Tray {
    $tray = Get-Process 'ollama app' -ErrorAction SilentlyContinue
    if ($tray) { $tray | Stop-Process -Force; Write-Log "Stopped the Ollama tray app." }
    $run = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    $props = Get-ItemProperty -Path $run -ErrorAction SilentlyContinue
    if ($props) {
        foreach ($p in $props.PSObject.Properties) {
            if ($p.Value -is [string] -and $p.Value -match 'ollama app') {
                Remove-ItemProperty -Path $run -Name $p.Name -Force
                Write-Log "Removed login autostart '$($p.Name)' (HKCU Run)."
            }
        }
    }
    $lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'Ollama.lnk'
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Log "Removed Startup shortcut: $lnk" }
}

# --- download + silent install -------------------------------------------
function Install-Ollama {
    $tmp = Join-Path $env:TEMP "OllamaSetup.exe"
    Write-Log "Downloading OllamaSetup.exe..."
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $tmp -UseBasicParsing -TimeoutSec 600
    Write-Log "Running installer (/VERYSILENT)..."
    $p = Start-Process -FilePath $tmp -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "OllamaSetup.exe exited with code $($p.ExitCode)." }
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

# --- health + runner verification ----------------------------------------
function Wait-ApiHealthy {
    param([int]$ApiPort, [int]$TimeoutSec = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/version" -TimeoutSec 5 | Out-Null
            return $true
        } catch { Start-Sleep -Seconds 2 }
    }
    return $false
}

# Prove the CUDA runner survived: load a tiny model and confirm it offloaded to GPU.
# size_vram == 0 is the CPU-only corruption (library=cpu / total_vram=0).
function Test-CudaRunner {
    param([int]$ApiPort)
    $base = "http://127.0.0.1:$ApiPort"
    $model = $VerifyModel
    if (-not $model) {
        try {
            $tags = Invoke-RestMethod -Uri "$base/api/tags" -TimeoutSec 15
            $smallest = $tags.models | Sort-Object size | Select-Object -First 1
            if ($smallest) { $model = $smallest.name }
        } catch { }
    }
    if (-not $model) { Write-Log "Runner check SKIPPED on :$ApiPort (no pulled model available)."; return "skipped" }

    Write-Log "Verifying CUDA runner on :$ApiPort with '$model'..."
    try {
        $loadBody = @{ model = $model; keep_alive = -1; stream = $false } | ConvertTo-Json
        Invoke-RestMethod -Uri "$base/api/generate" -Method Post -Body $loadBody -ContentType "application/json" -TimeoutSec 120 | Out-Null
        $ps = Invoke-RestMethod -Uri "$base/api/ps" -TimeoutSec 15
        $entry = $ps.models | Where-Object { $_.name -eq $model } | Select-Object -First 1
        $vram = if ($entry) { [int64]$entry.size_vram } else { 0 }
        # unload
        $unloadBody = @{ model = $model; keep_alive = 0; stream = $false } | ConvertTo-Json
        Invoke-RestMethod -Uri "$base/api/generate" -Method Post -Body $unloadBody -ContentType "application/json" -TimeoutSec 30 | Out-Null
        if ($vram -gt 0) { return "OK" } else { return "CPU-ONLY" }
    } catch {
        Write-Log "Runner check ERROR on :$ApiPort : $($_.Exception.Message)"
        return "error"
    }
}

# =========================================================================
# 1-3: version check + skip gate (no disruption)
# =========================================================================
$installed = Get-InstalledVersion
Write-Log "Installed Ollama version: $installed"

try {
    $latest = Get-LatestVersion
} catch {
    Write-Log "Could not reach GitHub for the latest version ($($_.Exception.Message)). Skipping this run."
    exit 0
}
Write-Log "Latest Ollama version: $latest"

if ($installed -eq $latest -and -not $Force) {
    Write-Log "$installed -> $latest | runner=n/a | already latest, no change."
    exit 0
}

# =========================================================================
# 4-11: the safe update sequence (always restore services + LLMConfig)
# =========================================================================
$runnerResult = "n/a"
$outcome = "unknown"
try {
    # 4. Stop LLMConfig so its ensure_running() can't Start-Service ollama mid-install.
    Write-Log "Stopping LLMConfig task '$LLMConfigTask'..."
    Stop-ScheduledTask -TaskName $LLMConfigTask -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $held = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $held) {
        Write-Log "Port $Port still held by PID $($conn.OwningProcess); stopping it."
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }

    # 5. Quiesce Ollama (both services + tray + strays) so the binary unlocks.
    Stop-OllamaProcesses

    # 6-7. Download + silent install.
    Install-Ollama

    # 8. Re-suppress the tray the installer just re-enabled.
    Suppress-Tray

    # 9. Restart Ollama and wait for each present instance's API.
    Start-OllamaServices
    foreach ($name in (Get-PresentServices)) {
        $apiPort = Get-ServicePort -Name $name
        if (Wait-ApiHealthy -ApiPort $apiPort) { Write-Log "'$name' API healthy on :$apiPort." }
        else { Write-Log "WARNING: '$name' API not healthy on :$apiPort within timeout." }
    }

    # 10. Verify the CUDA runner survived (the 2026-06-18 failure mode).
    $primary = (Get-PresentServices | Select-Object -First 1)
    if ($primary) {
        $apiPort = Get-ServicePort -Name $primary
        $runnerResult = Test-CudaRunner -ApiPort $apiPort
        if ($runnerResult -eq "CPU-ONLY") {
            Write-Log "*** RUNNER CPU-ONLY after update -- CUDA runner corrupted. Retrying reinstall once. ***"
            Stop-OllamaProcesses
            Install-Ollama
            Suppress-Tray
            Start-OllamaServices
            if (Wait-ApiHealthy -ApiPort $apiPort) {
                $runnerResult = Test-CudaRunner -ApiPort $apiPort
            }
            if ($runnerResult -ne "OK") {
                Write-Log "*** RUNNER STILL $runnerResult after retry -- MANUAL ATTENTION NEEDED (CPU-only fallback). ***"
            }
        }
    }

    $outcome = if ($runnerResult -eq "OK" -or $runnerResult -eq "skipped") { "updated OK" } else { "updated but runner $runnerResult" }
}
catch {
    $outcome = "FAILED: $($_.Exception.Message)"
    Write-Log "ERROR during update: $($_.Exception.Message)"
}
finally {
    # Always restore services + LLMConfig, even on mid-run failure.
    Start-OllamaServices
    Write-Log "Restarting LLMConfig task '$LLMConfigTask'..."
    Start-ScheduledTask -TaskName $LLMConfigTask -ErrorAction SilentlyContinue
}

# 12. Final log line.
$final = Get-InstalledVersion
Write-Log "$installed -> $final | runner=$runnerResult | $outcome"
if ($outcome -like "FAILED:*" -or $runnerResult -eq "CPU-ONLY" -or $runnerResult -eq "error") { exit 1 }
exit 0
