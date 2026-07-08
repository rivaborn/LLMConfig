# LLMConfig

A small **GPU-arbitrated control plane** that lets you pick *which model runs on the
GPU* and *on which server* — **Ollama** or **vLLM** — and guarantees the chosen model
is the **only** thing in VRAM, so it never spills to system RAM prematurely. One Web
UI + REST API + OpenAI-`/v1` gateway + CLI over both servers, across one or two GPUs.

Built for `192.168.1.40` (`Alien-3070-TI`, Windows 11) with an **RTX 3090 (24 GB)** as
the primary card and an optional **RTX 3070 Ti (8 GB)** as a second lane. It runs
Windows-native and reaches vLLM inside **WSL2**.

---

## The problem it solves

A 24 GB card holds **one big model at a time**, but two servers want it. Switching
"who owns the GPU" used to be a manual dance — unload Ollama with `keep_alive:0`,
`pkill` vLLM (or run `serve.sh`), and *hope* nothing was left resident or the next
model would VRAM-exhaust at startup. There was also no single answer to **what's
available, what's loaded, and on which server**.

LLMConfig automates that arbitration and answers those three questions. To load a
model it **evicts** the other server plus any other Ollama models, **waits until
`nvidia-smi` confirms the VRAM is actually freed**, and only then loads the target —
so it packs 100 % of VRAM before any CPU spill.

## Features

- **Pick + load** a model on Ollama or vLLM, GPU-arbitrated, via Web UI / REST / CLI.
- **Query** available models (Ollama tags + vLLM alias catalog), what's loaded, on
  which server, and live VRAM — per GPU lane.
- **Guaranteed packing:** evict → confirm-freed → load. Reports on-GPU vs on-CPU bytes
  and flags *premature* spill; `--max-pack` pushes `num_gpu` to fill VRAM first.
- **OpenAI `/v1` gateway** with **auto-load on first request** — a client's `/model`
  picker can switch models with no manual swap.
- **Two independent GPU lanes:** the 3090 can serve a big vLLM model while the 3070 Ti
  serves a small Ollama/vLLM model concurrently, with no cross-lane eviction.
- **Monitor tab:** live GPU thermals (core + hotspot + GDDR6X junction), power, VRAM,
  and Ollama GPU/CPU split, with a rolling history persisted across restarts.
- **Model management:** pull/delete Ollama models, edit the vLLM alias registry,
  trigger HuggingFace downloads — all as streamed jobs.
- **`doctor`:** read-only recon that verifies every on-box assumption before you trust
  a swap.

## Topology

|             | Ollama                 | vLLM                                   |
| ----------- | ---------------------- | -------------------------------------- |
| Runs in     | Windows 11 native      | WSL2 Ubuntu 24.04                      |
| Reach       | `127.0.0.1:11434`      | `127.0.0.1:11437` (socat relay)        |
| Model swap  | REST `keep_alive`      | one model/process — `serve.sh <alias>` |
| State via   | `/api/ps`              | relay `/v1/models`                     |

The control app itself listens on **`:11430`** (UI + REST + `/v1`). All endpoints are
**LAN-only** with no auth unless you set `LLMCONFIG_API_KEY`.

## How it works

```
                Web UI  ·  CLI  ·  REST  ·  OpenAI /v1
                                │
                    FastAPI (Windows-native, :11430)
                                │
                         Orchestrator
              ┌─────────────────┴──────────────────┐
          Lane: primary (RTX 3090)         Lane: companion (RTX 3070 Ti)
          ┌──────┴───────┐                 ┌──────┴───────┐
     OllamaBackend   VllmBackend      OllamaBackend   VllmBackend
     httpx :11434    relay :11437     httpx :11435    relay :11438
     Win service     wsl.exe →        NSSM service    wsl.exe →
     control         serve.sh /       control         serve-companion.sh
                     systemctl --user
          └──────┬───────┘                 └──────┬───────┘
     nvidia-smi (3090 by UUID)         nvidia-smi (3070 Ti by UUID)
     eviction-wait gate                eviction-wait gate
```

- A **Lane** binds one Ollama+vLLM pair to **one GPU, matched by UUID** (indices are
  unstable). Each lane arbitrates independently behind its own lock; lanes never evict
  each other.
- The **eviction-wait gate** is the core guarantee: evict everything else on the card,
  poll `nvidia-smi` until it's back to driver baseline, *then* load — 100 % VRAM before
  any spill.
- vLLM serves **one model per process**; a swap restarts the `vllm@<alias>` systemd
  unit (which runs `serve.sh`). vLLM status is read from the socat relay's `/v1/models`.
- The Orchestrator holds a **shared WSL keepalive** (`wsl.exe … sleep infinity`) so the
  distro — and any loaded vLLM model + relay — survives WSL2's idle-shutdown.

## Quickstart (on the box)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
copy .env.example .env                    # edit if any default differs from your box
.\.venv\Scripts\llmconfig doctor --local  # verify the box matches config
.\.venv\Scripts\llmconfig serve           # http://<box>:11430/
```

```bash
llmconfig status                          # GPU owner, loaded model, VRAM (every lane)
llmconfig models                          # Ollama tags + vLLM aliases
llmconfig load vllm coder30-awq           # swap the 3090 to a vLLM alias
llmconfig load ollama qwen3-coder:30b     # swap it to an Ollama model
llmconfig unload                          # free the GPU
```

Full deployment (systemd unit, Windows always-on task, firewall, companion lane):
see [`deploy/README-deploy.md`](deploy/README-deploy.md).

## Configuration

Every box-specific value lives in `.env` (gitignored; copy from `.env.example`). All
have sane defaults for `.40` in `llmconfig/config.py`.

| Variable                   | Default                     | Purpose                                             |
| -------------------------- | --------------------------- | --------------------------------------------------- |
| `LLMCONFIG_HOST` / `_PORT` | `0.0.0.0` / `11430`         | Where the control app listens                       |
| `LLMCONFIG_API_KEY`        | *(empty)*                   | If set, write ops require the `X-API-Key` header    |
| `OLLAMA_URL`               | `http://127.0.0.1:11434`    | Primary Ollama server                               |
| `OLLAMA_SERVICE_NAME`      | `ollama`                    | Windows service name (start/restart control)        |
| `VLLM_RELAY_URL`           | `http://127.0.0.1:11437`    | Primary vLLM socat relay (**use `127.0.0.1`**)      |
| `VLLM_SERVE_SCRIPT`        | `/home/folar/vllm/serve.sh` | The vLLM launcher inside WSL                        |
| `VLLM_SYSTEMD_UNIT`        | `vllm@`                     | Templated user unit; instance name = alias          |
| `WSL_DISTRO` / `WSL_USER`  | `Ubuntu-24.04` / `folar`    | The distro hosting vLLM                             |
| `GPU_UUID`                 | `GPU-739bece9-…` (RTX 3090) | Primary lane's card, **pinned by UUID**             |
| `VRAM_TOTAL_MB`            | `24576`                     | Primary card VRAM                                   |
| `VRAM_FREE_BASELINE_MB`    | `1500`                      | "VRAM is freed / GPU maxed" threshold               |
| `COMPANION_ENABLED`        | `false`                     | Turn on the RTX 3070 Ti lane                        |
| `COMPANION_*`              | *(3070 Ti defaults)*        | Companion GPU UUID, ports, service, relay, registry |
| `HF_TOKEN`                 | *(empty)*                   | HuggingFace token for vLLM downloads                |
| `MONITOR_ENABLED`          | `true`                      | Run the telemetry sampler                           |
| `MONITOR_INTERVAL_S`       | `5.0`                       | GPU sample cadence                                  |
| `MONITOR_RETENTION_H`      | `24`                        | History window (in-memory + on-disk)                |
| `MONITOR_PERSIST`          | `true`                      | Persist samples to SQLite (survives restart)        |

Timeouts (`EVICT_TIMEOUT_S`, `POLL_INTERVAL_S`, `DEFAULT_VLLM_LOAD_TIMEOUT_S`,
`VLLM_PROBE_TIMEOUT_S`, …) also live in `config.py` and are overridable via env.

## REST API

Interactive docs at `/docs`. Read/inference endpoints are open; **write** endpoints
require `X-API-Key` only when `LLMCONFIG_API_KEY` is set. Read endpoints take
`?lane=primary|companion` (default `primary`); load/unload take `lane` in the body.

| Method & path                           | Purpose                                                       |
| --------------------------------------- | ------------------------------------------------------------- |
| `GET /api/status`                       | Every lane under `lanes[]`: owner, loaded model, VRAM, swap   |
| `GET /api/lanes`                        | Configured lanes (id, name, enabled, current default)         |
| `GET /api/models?lane=`                 | That lane's Ollama tags + vLLM alias catalog (loaded flagged) |
| `GET /api/gpu?lane=`                    | Parsed `nvidia-smi` for that lane's GPU (by UUID)             |
| `GET /api/monitor`                      | Latest thermals/power/VRAM + Ollama split                     |
| `GET /api/monitor/history?window=`      | Bucketed telemetry history over the last `window` seconds     |
| `GET /api/doctor`                       | On-box recon report (per-lane checks)                         |
| `GET /api/jobs/{id}`                    | Progress + log for a long load/pull/download                  |
| `POST /api/load`                        | `{server,model,lane?,force?,max_pack?}` → a Job               |
| `POST /api/unload`                      | `{server?,lane?}` → free that lane's GPU                      |
| `GET / PUT /api/lanes/{id}/default`     | Get / set a lane's startup-default model                      |
| `POST /api/ollama/pull`                 | Pull an Ollama model (job)                                    |
| `DELETE /api/ollama/{name}`             | Delete an Ollama model                                        |
| `GET/POST/PUT/DELETE /api/vllm/aliases` | The vLLM alias registry for a lane (add/edit/remove)          |
| `POST /api/vllm/download`               | HuggingFace-download a model into the WSL cache               |

## OpenAI `/v1` gateway — auto-load on first request

An OpenAI-compatible gateway on the same port (`http://192.168.1.40:11430/v1`) so a
client can switch models **without a manual swap**: the first request for a model
loads it. Built for opencode's `/model` picker (which has no selection-time hook), so
the switch happens on the inference path.

`GET /v1/models` · `POST /v1/chat/completions` · `POST /v1/completions`.

- **Lane** = the `X-LLM-Lane` header (`primary` default; `companion` → the 3070 Ti).
- **Resolution:** a vLLM `served_name` (e.g. `qwen3-coder-30b`) → that lane's vLLM
  relay; else an Ollama tag (e.g. `qwen3-coder:30b`) → that lane's Ollama; else `404`.
- **No new arbitration** — it reuses `/api/load` (the per-lane lock, eviction-wait
  gate, WSL keepalive). On a **streaming** request it relays the cold-load progress as
  chat chunks (`⏳ …`), then forwards the real completion verbatim.
- **Edge cases:** identical concurrent loads coalesce onto one job; a non-stream
  request that arrives mid-load of a *different* model returns an empty `200` (so
  title-gen never blocks); cold-load timeout → `503`. LAN-only, like the rest.

Point a provider's `baseURL` at `http://192.168.1.40:11430/v1`. The always-on app must
be **restarted** to pick up a new gateway build.

## CLI

The `llmconfig` command is a thin client over the REST API (plus `serve` to launch it).

```bash
llmconfig status                              # every lane: owner, VRAM, loaded model
llmconfig models --lane companion             # Ollama tags + vLLM aliases for a lane
llmconfig gpu                                 # nvidia-smi state for a lane's GPU
llmconfig load vllm coder30-awq               # swap the PRIMARY 3090 to a vLLM alias
llmconfig load --lane companion ollama qwen3:4b   # load a model on the 3070 Ti
llmconfig load ollama qwen3-coder:30b --max-pack  # fill VRAM before spilling
llmconfig unload --lane companion             # free the companion GPU
llmconfig companion-default ollama qwen3:4b   # set a lane's auto-load-on-startup default
llmconfig pull qwen3:4b                        # pull an Ollama model
llmconfig doctor                               # verify the box (add --local to run in-process)
llmconfig serve                                # run the API + web UI here
```

Point it at the box with `--url http://192.168.1.40:11430` or `$LLMCONFIG_URL`; pass
`--api-key` / `$LLMCONFIG_API_KEY` when auth is on.

## Monitor (telemetry)

The Monitor tab (and `/api/monitor*`) sample every visible GPU every `MONITOR_INTERVAL_S`
seconds: **core temp** + power + utilization + VRAM from `nvidia-smi`, plus **hotspot
and GDDR6X memory-junction temps via NVAPI** (nvidia-smi returns `N/A` for those on
consumer GeForce cards), plus the primary lane's Ollama GPU-vs-CPU split. Samples land
in rolling in-memory deques **and** a best-effort SQLite DB (`data/monitor.db`), so the
history window survives an app/service restart. Persistence failures degrade to
in-memory only — they never take down the sampler.

## Idle auto-unload (power saving)

A resident model pins the card in the **P0** power state — memory clocks never drop, so
the 3090 draws **~117 W doing nothing** instead of its ~25 W **P8** idle. Neither server
lets go on its own (LLMConfig loads Ollama with `keep_alive:-1`; vLLM never
auto-unloads), so a background **idle reaper** (`llmconfig/idle.py`, on by default)
unloads a lane after `IDLE_UNLOAD_AFTER_MIN` minutes (default 15) with no observed
activity, letting the card fall to P8.

**Activity** is any of: a `/v1` gateway request routed to the lane, a load finishing, or
a Monitor **utilization sample above `IDLE_UNLOAD_UTIL_PCT`** (default 5 %) on the
lane's GPU — the util signal catches clients that hit Ollama or the vLLM relay directly,
bypassing the gateway. Each lane's seconds-since-activity is reported as `idle_s` in
`GET /api/status` → `lanes[]`.

Reaping goes through the same per-lane unload path as `POST /api/unload` (lane lock +
eviction-wait gate), and a reaped model returns hands-free: the next `/v1` request
auto-loads it (direct-Ollama clients reload through Ollama itself). When no lane serves
vLLM anymore the reaper also releases the WSL keepalive so the WSL2 distro can
idle-shutdown; the next vLLM load restarts it. Set `IDLE_UNLOAD_ENABLED=false` to keep
models pinned. If the lane's GPU also renders a desktop, background compositing can
register as activity — raise `IDLE_UNLOAD_UTIL_PCT`.

## Context size (Ollama)

`/api/load` (and the `/v1` auto-load) do **not** take a context-length parameter — an
Ollama model loads at the `num_ctx` baked into its Modelfile, so the KV-cache VRAM
footprint is fixed by the *tag* you load, not the caller. To run a smaller context
(to leave VRAM headroom on the 24 GB card), bake a context-specific tag:

```bash
# Modelfile:  FROM <model>  +  PARAMETER num_ctx 65536
ollama create <model>-64k -f Modelfile    # reuses the existing weights blob
llmconfig load ollama <model>-64k
```

Also note Ollama's global default `OLLAMA_CONTEXT_LENGTH=4096` **silently truncates**
every model — raise it on the service (see the deploy guide) so served context clears
your workloads.

## vLLM alias registry

vLLM's `/v1/models` only reports the *currently-served* model, so the set of models
vLLM *can* serve is an editable **registry** (`data/vllm_models.yaml`, seeded from the
package default; companion lane uses `vllm_models_companion.yaml`). Each entry maps an
`alias` → a `served_name`, HF repo, mode, status, and notes.

> **Adding a new vLLM model takes a `serve.sh` case, not just a registry row.** The
> registry's `launch_args`/`managed_by: registry` fields are currently unwired —
> `_load_vllm` always launches via `vllm@<alias>` → `serve.sh <alias>`, whose hardcoded
> `case` sets the launch args. So: (1) add a `case` to `deploy/serve.sh` (commit it),
> (2) `POST /api/vllm/aliases` with `managed_by: serve.sh`, (3) download the repo,
> (4) `llmconfig load vllm <alias>`.

## Deployment

Windows-native app + WSL2 vLLM. The [`deploy/`](deploy/) directory has everything:

- `install-service.ps1` — always-on Scheduled Task `LLMConfig` (logon, elevated) + firewall.
- `install-companion.ps1` — the 2nd Ollama (`OllamaCompanion`, 3070 Ti, `:11435`).
- `vllm@.service` / `vllm-companion@.service` — the templated systemd-user units.
- `serve.sh` — the vendored vLLM launcher (per-alias args + torch-based GPU pinning).
- `update-ollama.ps1` / `install-ollama-update.ps1` — safe (CUDA-runner-verifying)
  Ollama updater + a weekly task (the tray auto-updater corrupts the NSSM install).

Full step-by-step, including the companion lane and all the live-tested GPU-pinning
gotchas, is in [`deploy/README-deploy.md`](deploy/README-deploy.md).

## Troubleshooting / gotchas

- **vLLM died seconds after loading** → WSL2 idle-shut-down the distro. The app holds a
  keepalive for its lifetime; this only happens if the app was killed (not stopped) —
  `wsl --shutdown` clears an orphaned keepalive.
- **`/api/status` is slow / hangs** → a *down* vLLM relay blackholes the SYN under WSL2
  localhost-forwarding; `VLLM_PROBE_TIMEOUT_S` (1 s) caps it. Never use `localhost` for
  the relay — `127.0.0.1` avoids the IPv4/IPv6 happy-eyeballs delay.
- **Ollama silently running on CPU (`library=cpu`)** → the tray auto-updater corrupted
  the CUDA runner. Reinstall via `deploy/update-ollama.ps1`; never run `ollama app.exe`.
- **"Access denied" on load** → the app isn't elevated and can't `Restart-Service ollama`.
  Run it as the elevated Scheduled Task / a LocalSystem service.
- **New app instance wedges on `:11430`** → `Stop-ScheduledTask` left the old uvicorn
  child holding the port; kill it before `Start-ScheduledTask`.
- **A model loads on the wrong card** → a GPU was pinned by index somewhere. Everything
  pins by UUID; `serve.sh` resolves the vLLM index via torch (vLLM ignores
  `CUDA_DEVICE_ORDER` and uses FASTEST_FIRST order). Run `llmconfig doctor`.

## Project layout

```
llmconfig/
  config.py         Settings + LaneConfig (all box-specific values)
  main.py           FastAPI app: REST endpoints + static UI + lifespan
  orchestrator.py   coordinates lanes; shared WSL keepalive + defaults
  lane.py           per-GPU arbitration state machine (eviction-wait gate)
  lane_state.py     persisted per-lane default model
  backends/
    ollama.py       Ollama REST client + Windows service control
    vllm.py         vLLM relay status + serve.sh/systemctl lifecycle over wsl.exe
  gpu.py            nvidia-smi truth (by UUID) + Monitor metric sampling
  nvapi.py          NVAPI hotspot + GDDR6X junction temps (ctypes)
  monitor.py        telemetry sampler + SQLite history
  idle.py           idle auto-unload policy (reap an unused lane → GPU drops to P8)
  registry.py       vLLM alias catalog (YAML)
  schemas.py        pydantic models
  jobs.py           async job manager (streamed logs)
  wsl.py            wsl.exe bridge + WslKeepalive
  winsvc.py         Windows service control
  proc.py           subprocess wrapper
  doctor.py         read-only on-box recon
  openai_gateway.py OpenAI /v1 gateway (auto-load)
  cli.py            the `llmconfig` CLI
  web/              static UI + templates
  data/*.default.yaml   shipped registry seeds
deploy/             install scripts, serve.sh, systemd units, deploy guide
tests/              pytest (respx-mocked; no GPU needed)
data/               live registry, lane defaults, monitor.db (gitignored)
```

## Testing

```bash
pip install -e ".[dev]" && pytest
```

Unit tests mock Ollama/vLLM HTTP (`respx`) and stub `nvidia-smi`/`wsl.exe`, so they run
anywhere — **no GPU required**. `asyncio_mode = auto` (see `pyproject.toml`).

## Status

**Live-verified on `.40`.** `doctor --local` green; both paths exercised end-to-end on
the RTX 3090 (Ollama load/unload, vLLM load that evicts → packs VRAM → serves through
the relay → unloads to 0 MiB). The companion 3070 Ti lane is proven for Ollama;
companion vLLM is optional and installed separately. Telemetry is persisted across
restarts. See the homelab wiki (`hosts/ollama-host/services/llmconfig`) for the
deployed-state details.

## License

MIT.
