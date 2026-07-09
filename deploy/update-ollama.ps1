<#
Safely update Ollama on .40 (both NSSM instances) on a schedule, then verify each
instance's runner (a) offloads to GPU at all and (b) lands on the GPU its service is
PINNED to. Replaces the manual "Stop-Service -> OllamaSetup -> Start-Service" dance
documented in install-companion.ps1.

The pinned-GPU check exists because an Ollama update can silently break pinning
without breaking GPU offload: the 0.19 -> 0.30 update (2026-06-21) added Vulkan GPU
discovery that ignores CUDA_VISIBLE_DEVICES, and the companion's model quietly moved
onto the 3090 (found 2026-07-08 as ~2 GB of "unowned" VRAM holding the card in P0 at
~117 W). The old "size_vram > 0" check passed the whole time. See the GPU pinning
note in install-companion.ps1 for the env fix (OLLAMA_VULKAN=0 + UUID pin).

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
    [string]$RepoPath = "",
    [string[]]$ServiceNames = @('Ollama', 'OllamaCompanion'),
    [string]$LLMConfigTask = "LLMConfig",
    [int]$Port = 11430,                 # LLMConfig's control port (freed if still held)
    [string]$OllamaExe = "",
    [string]$VerifyModel = "",          # empty => auto-pick the smallest already-pulled model
    [switch]$Force,                     # reinstall even when already on the latest version
    [string]$LogPath = ""
)
$ErrorActionPreference = "Stop"
# $PSScriptRoot is unreliable under Task Scheduler -File on this box (it came back empty,
# which made RepoPath resolve to C:\ and logs land in C:\logs). Resolve robustly; the
# scheduled task also passes -RepoPath explicitly (see install-ollama-update.ps1).
if (-not $RepoPath) {
    if ($PSScriptRoot)      { $RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
    elseif ($PSCommandPath) { $RepoPath = (Resolve-Path (Join-Path (Split-Path $PSCommandPath) "..")).Path }
    else                    { $RepoPath = (Get-Location).Path }
}
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

# The GPU a service is pinned to, as a UUID. Reads CUDA_VISIBLE_DEVICES from the
# service's NSSM env; a legacy *index* pin (pre 2026-07-08) is translated via
# nvidia-smi, whose enumeration is PCI order -- the same order those pins were
# defined under (CUDA_DEVICE_ORDER=PCI_BUS_ID). $null = unpinned / unknowable.
function Get-ServicePinnedUuid {
    param([string]$Name)
    $nssm = (Get-Command nssm.exe -ErrorAction SilentlyContinue).Source
    if (-not $nssm) { return $null }
    try { $extra = (& $nssm get $Name AppEnvironmentExtra 2>$null) | Out-String } catch { return $null }
    $m = [regex]::Match($extra, 'CUDA_VISIBLE_DEVICES=(GPU-[0-9a-fA-F-]+|\d+)')
    if (-not $m.Success) { return $null }
    $pin = $m.Groups[1].Value
    if ($pin -like 'GPU-*') { return $pin }
    try {
        $line = (nvidia-smi --query-gpu=index,uuid --format=csv,noheader) |
            Where-Object { ($_ -split ',')[0].Trim() -eq $pin } | Select-Object -First 1
        if ($line) { return ($line -split ',')[1].Trim() }
    } catch { }
    return $null
}

# Per-GPU memory.used (MiB) keyed by UUID. $null when nvidia-smi is unavailable --
# the pin check then degrades to unverified rather than failing the run.
function Get-GpuMemSnapshot {
    try {
        $snap = @{}
        foreach ($line in (nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits)) {
            $parts = $line -split ','
            $snap[$parts[0].Trim()] = [int]$parts[1].Trim()
        }
        if ($snap.Count) { return $snap }
    } catch { }
    return $null
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
    # Use -PassThru + WaitForExit, NOT -Wait: the installer auto-launches the persistent tray
    # ('ollama app') + a server ('ollama'), and Start-Process -Wait blocks on the whole process
    # tree -> it hangs forever (verified live 2026-06-21). WaitForExit() waits only for the
    # installer process, which exits once it has spawned the app.
    $p = Start-Process -FilePath $tmp -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -PassThru
    $null = $p.Handle   # cache handle so ExitCode is readable after exit (PS -PassThru gotcha)
    if (-not $p.WaitForExit(300000)) { throw "OllamaSetup.exe did not exit within 5 minutes." }
    if ($p.ExitCode -ne 0) { throw "OllamaSetup.exe exited with code $($p.ExitCode)." }
    # The installer auto-launches the tray + a server that squat :11434; kill them so the NSSM
    # services can rebind. (Suppress-Tray below also handles the tray + its autostart key.)
    Write-Log "Installer done; killing the tray + stray server it auto-launched."
    Get-Process 'ollama app', 'ollama' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
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

# Prove the runner survived: load a tiny model, confirm it offloaded to GPU
# (size_vram == 0 is the CPU-only corruption: library=cpu / total_vram=0), and
# confirm the VRAM appeared on the service's PINNED card (per-GPU memory.used
# delta around the load, by UUID -- Ollama can't report which card it used).
# Deltas are clean because the services were just restarted, so nothing of ours
# is resident when the baseline is taken. Returns OK / CPU-ONLY / WRONG-GPU /
# skipped / error.
function Test-CudaRunner {
    param([int]$ApiPort, [string]$ServiceName)
    $base = "http://127.0.0.1:$ApiPort"
    $model = $VerifyModel
    if (-not $model) {
        try {
            $tags = Invoke-RestMethod -Uri "$base/api/tags" -TimeoutSec 15
            $smallest = $tags.models | Sort-Object size | Select-Object -First 1
            if ($smallest) { $model = $smallest.name }
        } catch { }
    }
    if (-not $model) { Write-Log "Runner check SKIPPED for '$ServiceName' on :$ApiPort (no pulled model available)."; return "skipped" }

    $pin = Get-ServicePinnedUuid -Name $ServiceName
    $before = Get-GpuMemSnapshot
    Write-Log "Verifying runner for '$ServiceName' on :$ApiPort with '$model' (pin: $(if ($pin) { $pin } else { 'none/unknown' }))..."
    try {
        $loadBody = @{ model = $model; keep_alive = -1; stream = $false } | ConvertTo-Json
        Invoke-RestMethod -Uri "$base/api/generate" -Method Post -Body $loadBody -ContentType "application/json" -TimeoutSec 120 | Out-Null
        $ps = Invoke-RestMethod -Uri "$base/api/ps" -TimeoutSec 15
        $entry = $ps.models | Where-Object { $_.name -eq $model } | Select-Object -First 1
        $vram = if ($entry) { [int64]$entry.size_vram } else { 0 }
        $after = Get-GpuMemSnapshot

        # Which card actually gained the VRAM? (>= 200 MiB = attributable; the
        # smallest pulled model is ~1 GB, so a real load always clears this.)
        $pinResult = "unverified"
        if ($vram -gt 0 -and $pin -and $before -and $after) {
            $bestUuid = $null; $bestDelta = 0
            foreach ($uuid in $after.Keys) {
                $b = if ($before.ContainsKey($uuid)) { $before[$uuid] } else { 0 }
                $delta = $after[$uuid] - $b
                if ($delta -gt $bestDelta) { $bestDelta = $delta; $bestUuid = $uuid }
            }
            if ($bestDelta -ge 200 -and $bestUuid -ne $pin) {
                Write-Log "*** '$ServiceName' loaded '$model' on the WRONG GPU: +$bestDelta MiB on $bestUuid, pinned to $pin. ***"
                $pinResult = "wrong"
            } elseif ($bestDelta -ge 200) {
                Write-Log "'$ServiceName': +$bestDelta MiB on the pinned GPU -- pin verified."
                $pinResult = "ok"
            } else {
                Write-Log "'$ServiceName': VRAM delta inconclusive (<200 MiB on every card) -- pin unverified."
            }
        }

        # unload
        $unloadBody = @{ model = $model; keep_alive = 0; stream = $false } | ConvertTo-Json
        Invoke-RestMethod -Uri "$base/api/generate" -Method Post -Body $unloadBody -ContentType "application/json" -TimeoutSec 30 | Out-Null
        if ($vram -le 0) { return "CPU-ONLY" }
        if ($pinResult -eq "wrong") { return "WRONG-GPU" }
        return "OK"
    } catch {
        Write-Log "Runner check ERROR for '$ServiceName' on :$ApiPort : $($_.Exception.Message)"
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

    # 4b. Free the primary GPU. vLLM runs in WSL and does NOT exit when LLMConfig stops,
    #     so the CUDA-runner check (step 10) would load Ollama onto an already-full 3090
    #     and mis-read it as CPU-only. WSL here hosts only vLLM; LLMConfig restarts it in
    #     the finally block. (Verified live 2026-06-21: WSL/vLLM persisted and the 3090
    #     stayed at ~23.7 GiB after the task stopped LLMConfig.)
    Write-Log "Shutting down WSL to release the primary GPU (vLLM)..."
    & wsl.exe --shutdown 2>$null
    Start-Sleep -Seconds 5

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

    # 10. Verify EACH instance's runner: it must offload to GPU (the 2026-06-18
    #     corruption) AND land on its pinned card (the 2026-07-08 Vulkan-discovery
    #     regression -- that one hit the companion, so checking only the primary
    #     is not enough).
    $results = [ordered]@{}
    $reinstalled = $false
    foreach ($name in (Get-PresentServices)) {
        $apiPort = Get-ServicePort -Name $name
        $r = Test-CudaRunner -ApiPort $apiPort -ServiceName $name
        if ($r -eq "CPU-ONLY" -and -not $reinstalled) {
            # Corruption is install-level: one reinstall retry can fix it.
            Write-Log "*** '$name' runner CPU-ONLY after update -- CUDA runner corrupted. Retrying reinstall once. ***"
            $reinstalled = $true
            Stop-OllamaProcesses
            Install-Ollama
            Suppress-Tray
            Start-OllamaServices
            if (Wait-ApiHealthy -ApiPort $apiPort) {
                $r = Test-CudaRunner -ApiPort $apiPort -ServiceName $name
            }
            if ($r -ne "OK") {
                Write-Log "*** '$name' runner STILL $r after retry -- MANUAL ATTENTION NEEDED (CPU-only fallback). ***"
            }
        }
        if ($r -eq "WRONG-GPU") {
            # Pin bypass is env/discovery-level: a reinstall can't fix it, so don't retry.
            Write-Log "*** '$name' is IGNORING its GPU pin -- a reinstall won't fix this; check the service env (OLLAMA_VULKAN=0 + GGML_VK_VISIBLE_DEVICES=-1 + UUID pin, see install-companion.ps1). MANUAL ATTENTION NEEDED. ***"
        }
        $results[$name] = $r
    }
    $runnerResult = if ($results.Count) {
        ($results.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join " "
    } else { "n/a" }

    $bad = @($results.Values | Where-Object { $_ -notin @("OK", "skipped") })
    $outcome = if (-not $bad.Count) { "updated OK" } else { "updated but runner problems: $($bad -join ', ')" }
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
if ($outcome -like "FAILED:*" -or $runnerResult -match "CPU-ONLY|WRONG-GPU|error") { exit 1 }
exit 0
