<#
Install the COMPANION Ollama instance for LLMConfig's 3070 Ti lane.

Registers a second Ollama as a Windows service ("OllamaCompanion") via NSSM, pinned
to the RTX 3070 Ti, listening on its own port (11435), and sharing the primary
Ollama's model store (so models pulled on either instance are visible to both).
LLMConfig then controls it through the same winsvc path as the primary
(Get-/Start-/Restart-Service OllamaCompanion).

GPU pinning note (learned live on this box): Ollama's CUDA_VISIBLE_DEVICES wants a
device *index*, not a `GPU-<uuid>` string (it won't resolve the UUID and falls back to
CPU). We keep the UUID as the source of truth and translate it to an index under
CUDA_DEVICE_ORDER=PCI_BUS_ID (so indices match nvidia-smi). The NSSM service runs as
LocalSystem and sets CUDA_VISIBLE_DEVICES in its own AppEnvironmentExtra, so it pins
the 3070 Ti explicitly. (A User-scope CUDA_VISIBLE_DEVICES=1 exists for the 3090 but
only affects user-session/WSL processes; LocalSystem services don't inherit it.)

Bind: by default the companion Ollama binds LAN-wide (0.0.0.0:11435) and a firewall
rule is added, so an off-box client (e.g. the opencode /swap relay) can reach the
3070 Ti directly. Like the primary Ollama it is **auth-less and LAN-only** -- never
expose it past the perimeter. Pass -OnBoxOnly to bind 127.0.0.1 and skip the firewall.

By default this also stops the Ollama tray app and removes its login-autostart: the
tray hosts the auto-updater, which can't stop the NSSM-managed ollama.exe and corrupts
the install on update (rollback wipes the CUDA runner -> CPU-only) and fights the
service for :11434. Pass -KeepTrayApp to leave it alone. After this, update Ollama
manually: Stop-Service Ollama -> run OllamaSetup -> Start-Service Ollama.

Run elevated:
    powershell -ExecutionPolicy Bypass -File deploy\install-companion.ps1

After this: set COMPANION_ENABLED=1 in .env and run `llmconfig doctor --local` - the
`companion.ollama.*` and `companion.gpu` checks should go green. The companion vLLM
lane (serve-companion.sh + a 2nd socat relay + vllm-companion@.service) is a separate,
optional step - see deploy/README-deploy.md.
#>
[CmdletBinding()]
param(
    [string]$ServiceName = "OllamaCompanion",
    [int]$Port = 11435,
    [string]$GpuUuid = "GPU-2caf7863-102e-31e5-be4d-5ec860addc78",  # the RTX 3070 Ti (source of truth)
    [int]$GpuIndex = -1,            # override; otherwise derived from $GpuUuid
    [string]$OllamaExe = "",
    [string]$ModelsDir = "",
    [switch]$KeepTrayApp,           # by default, stop + disable the Ollama tray/updater
    [switch]$OnBoxOnly             # bind 127.0.0.1 (no LAN/firewall); default is LAN 0.0.0.0 + firewall
)
$ErrorActionPreference = "Stop"

# --- Quiesce the Ollama tray app / auto-updater (unless -KeepTrayApp) ---
# The tray (`ollama app.exe`) hosts Ollama's auto-updater, which can't stop the
# NSSM-managed ollama.exe and so corrupts the install on update (DeleteFile
# access-denied -> rollback wipes the CUDA runner -> CPU-only). It also fights the
# service for :11434. A headless NSSM deployment (primary + companion) doesn't need it.
if (-not $KeepTrayApp) {
    $tray = Get-Process 'ollama app' -ErrorAction SilentlyContinue
    if ($tray) { $tray | Stop-Process -Force; Write-Host "Stopped the Ollama tray app." }
    else { Write-Host "Ollama tray app not running." }
    # remove login autostart so it can't relaunch (match by the value's data, not its name)
    $run = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
    $props = Get-ItemProperty -Path $run -ErrorAction SilentlyContinue
    if ($props) {
        foreach ($p in $props.PSObject.Properties) {
            if ($p.Value -is [string] -and $p.Value -match 'ollama app') {
                Remove-ItemProperty -Path $run -Name $p.Name -Force
                Write-Host "Removed login autostart '$($p.Name)' (HKCU Run)."
            }
        }
    }
    $lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'Ollama.lnk'
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Host "Removed Startup shortcut: $lnk" }
}

$nssm = (Get-Command nssm.exe -ErrorAction SilentlyContinue).Source
if (-not $nssm) { throw "NSSM not found on PATH. Install it (e.g. 'choco install nssm'), then re-run." }

if (-not $OllamaExe) {
    $cmd = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($cmd) { $OllamaExe = $cmd.Source }
    elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") { $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
    else { throw "ollama.exe not found. Pass -OllamaExe <path>." }
}

# Share the primary instance's model store so we don't double-download.
if (-not $ModelsDir) {
    if ($env:OLLAMA_MODELS) { $ModelsDir = $env:OLLAMA_MODELS }
    else { $ModelsDir = Join-Path $env:USERPROFILE ".ollama\models" }
}

# Translate the UUID -> device index under PCI_BUS_ID order (Ollama needs an index).
if ($GpuIndex -lt 0) {
    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    $line = (nvidia-smi --query-gpu=index,uuid --format=csv,noheader) | Where-Object { $_ -match [regex]::Escape($GpuUuid) }
    if (-not $line) { throw "GPU UUID $GpuUuid not found by nvidia-smi. Pass -GpuIndex explicitly." }
    $GpuIndex = [int](($line -split ",")[0].Trim())
}

$BindHost = if ($OnBoxOnly) { "127.0.0.1" } else { "0.0.0.0" }

Write-Host "ollama.exe : $OllamaExe"
Write-Host "models dir : $ModelsDir  (shared with the primary instance)"
Write-Host "GPU pin    : index $GpuIndex  (= $GpuUuid, PCI_BUS_ID order)"
Write-Host "listen     : ${BindHost}:$Port  $(if ($OnBoxOnly) {'(on-box only)'} else {'(LAN + firewall)'})"

if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Service '$ServiceName' already exists - reconfiguring."
    & $nssm stop $ServiceName | Out-Null
} else {
    & $nssm install $ServiceName $OllamaExe serve
}
& $nssm set $ServiceName AppDirectory (Split-Path $OllamaExe)
& $nssm set $ServiceName Start SERVICE_AUTO_START
& $nssm set $ServiceName DisplayName "Ollama (companion, RTX 3070 Ti)"
& $nssm set $ServiceName AppEnvironmentExtra `
    "OLLAMA_HOST=${BindHost}:$Port" `
    "CUDA_DEVICE_ORDER=PCI_BUS_ID" `
    "CUDA_VISIBLE_DEVICES=$GpuIndex" `
    "OLLAMA_MODELS=$ModelsDir"
& $nssm start $ServiceName
Write-Host "Started '$ServiceName' (Ollama on ${BindHost}:$Port pinned to GPU index $GpuIndex / the 3070 Ti)."

# LAN access for off-box clients (e.g. the opencode /swap relay). Auth-less + LAN-only.
if (-not $OnBoxOnly) {
    $fwName = "OllamaCompanion $Port"
    if (-not (Get-NetFirewallRule -DisplayName $fwName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $fwName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
        Write-Host "Added firewall rule '$fwName' (inbound TCP $Port)."
    } else {
        Write-Host "Firewall rule '$fwName' already present."
    }
}

Write-Host "`nVerify GPU offload in the service log (should NOT say library=cpu / total_vram=0):"
Write-Host "  nssm get $ServiceName AppStdout   # then load a small model and check nvidia-smi"
Write-Host "Next: set COMPANION_ENABLED=1 in .env, then: llmconfig doctor --local"
