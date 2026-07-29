<#
Watchdog for the .40 config surface — is `.env` still there, and is it still
being READ?

Why this exists: on 2026-07-24 22:11:28 Bitdefender Active Threat Control
quarantined this repo's `.env` (Atc4.Detection) along with its backup. Nothing
announced it; the loss was discovered days later, by which time the box had been
running on three Windows user environment variables instead. AV exceptions were
added and `.env` recreated on 2026-07-29, but ATC is BEHAVIOURAL — it fires on
what processes do, not on a file's name — so an exception is a hypothesis until
a week of real startups has tested it.

The unit count is the load-bearing check, not the file's existence. The Sparks
and the companion lane exist ONLY if `.env` (or the env vars) supplied
SPARK_ENABLED / COMPANION_ENABLED, so "six units" is proof the config was
actually parsed. A present-but-ignored file would still read as six today
because the env vars duplicate it — that ambiguity resolves itself the moment
those vars are removed (see DeferedWikiUpdates.md §E, not before 2026-08-05).

Deliberately quiet: it writes a line ONLY when the state CHANGES, so a week of
health leaves a two-line log rather than 672 identical entries. Exit code is 1
while in an alert state, so Task Scheduler's Last Run Result is a second signal.

    powershell -ExecutionPolicy Bypass -File deploy\watch-config.ps1

`-NtfyUrl` is a hook, not a default: at the time of writing there was no
reachable ntfy (the known server is at Site B, and 192.168.1.30 refuses). Pass
one once a Site A channel exists.
#>
[CmdletBinding()]
param(
    [string]$EnvPath       = "C:\Coding\rivaborn\LLMConfig\.env",
    [string]$ApiUrl        = "http://127.0.0.1:11430/api/status",
    [int]   $ExpectedUnits = 6,
    [string]$StateDir      = "C:\Coding\rivaborn\LLMConfig\data",
    [string]$NtfyUrl       = ""
)

$ErrorActionPreference = "Stop"
$stamp   = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$logFile = Join-Path $StateDir "watch-config.log"
$stFile  = Join-Path $StateDir "watch-config.state"
if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Force -Path $StateDir | Out-Null }

# --- observe -------------------------------------------------------------- #
$envOk = Test-Path $EnvPath
$units = -1
$apiErr = ""
try {
    $r = Invoke-RestMethod -Uri $ApiUrl -TimeoutSec 25
    $units = @($r.lanes).Count
} catch {
    $apiErr = $_.Exception.Message
}

$healthy = $envOk -and ($units -eq $ExpectedUnits)
$state   = "env={0} units={1}" -f $envOk, $units
$detail  = if ($apiErr) { "$state api_error='$apiErr'" } else { $state }

# --- compare with the last observation ------------------------------------ #
$previous = ""
if (Test-Path $stFile) { $previous = (Get-Content $stFile -Raw).Trim() }

if ($state -ne $previous) {
    $tag  = if ($healthy) { "OK     " } else { "ALERT  " }
    $line = "$stamp  $tag $detail"
    if ($previous) { $line += "   (was: $previous)" }
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Set-Content -Path $stFile -Value $state -Encoding utf8

    if ($NtfyUrl -and -not $healthy) {
        # Best effort only: a failed notification must not mask the real finding,
        # which is already durably in the log.
        try {
            Invoke-RestMethod -Uri $NtfyUrl -Method Post -TimeoutSec 15 `
                -Headers @{ Title = "LLMConfig .40 config change"; Priority = "5"; Tags = "warning" } `
                -Body $line | Out-Null
        } catch { }
    }
}

Write-Output "$stamp  $(if ($healthy) { 'healthy' } else { 'ALERT' })  $detail"
if (-not $healthy) { exit 1 }
exit 0
