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
CUDA_DEVICE_ORDER=PCI_BUS_ID (so indices match nvidia-smi). This box also sets a
User-scope CUDA_VISIBLE_DEVICES=1 (the 3090) as the default pin; the NSSM service runs
as LocalSystem and does NOT inherit that, and AppEnvironmentExtra sets the 3070 Ti
explicitly — so the companion lands on the right card.

Run elevated:
    powershell -ExecutionPolicy Bypass -File deploy\install-companion.ps1

After this: set COMPANION_ENABLED=1 in .env and run `llmconfig doctor --local` — the
`companion.ollama.*` and `companion.gpu` checks should go green. The companion vLLM
lane (serve-companion.sh + a 2nd socat relay + vllm-companion@.service) is a separate,
optional step — see deploy/README-deploy.md.
#>
[CmdletBinding()]
param(
    [string]$ServiceName = "OllamaCompanion",
    [int]$Port = 11435,
    [string]$GpuUuid = "GPU-2caf7863-102e-31e5-be4d-5ec860addc78",  # the RTX 3070 Ti (source of truth)
    [int]$GpuIndex = -1,            # override; otherwise derived from $GpuUuid
    [string]$OllamaExe = "",
    [string]$ModelsDir = ""
)
$ErrorActionPreference = "Stop"

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

Write-Host "ollama.exe : $OllamaExe"
Write-Host "models dir : $ModelsDir  (shared with the primary instance)"
Write-Host "GPU pin    : index $GpuIndex  (= $GpuUuid, PCI_BUS_ID order)"
Write-Host "listen     : 127.0.0.1:$Port"

if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Service '$ServiceName' already exists — reconfiguring."
    & $nssm stop $ServiceName | Out-Null
} else {
    & $nssm install $ServiceName $OllamaExe serve
}
& $nssm set $ServiceName AppDirectory (Split-Path $OllamaExe)
& $nssm set $ServiceName Start SERVICE_AUTO_START
& $nssm set $ServiceName DisplayName "Ollama (companion, RTX 3070 Ti)"
& $nssm set $ServiceName AppEnvironmentExtra `
    "OLLAMA_HOST=127.0.0.1:$Port" `
    "CUDA_DEVICE_ORDER=PCI_BUS_ID" `
    "CUDA_VISIBLE_DEVICES=$GpuIndex" `
    "OLLAMA_MODELS=$ModelsDir"
& $nssm start $ServiceName
Write-Host "Started '$ServiceName' (Ollama on :$Port pinned to GPU index $GpuIndex / the 3070 Ti)."
Write-Host "`nVerify GPU offload in the service log (should NOT say library=cpu / total_vram=0):"
Write-Host "  nssm get $ServiceName AppStdout   # then load a small model and check nvidia-smi"
Write-Host "Next: set COMPANION_ENABLED=1 in .env, then: llmconfig doctor --local"
