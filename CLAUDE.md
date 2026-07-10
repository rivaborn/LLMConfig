# CLAUDE.md — working in the LLMConfig repo

Guidance for Claude Code (and any coding agent) editing this project. Read this
before making changes; it captures the architecture, the hard invariants, and the
non-obvious gotchas that live testing on the box already paid for.

## What this is

A **GPU-arbitrated control plane for Ollama + vLLM** on a single box that has one or
two NVIDIA GPUs. It guarantees that the chosen model is the **sole occupant** of a
card (100 % of VRAM before any CPU spill) by evicting whatever else is on that GPU
and *waiting until `nvidia-smi` confirms the VRAM is actually freed* before loading.
It wraps Ollama and vLLM behind one Web UI + REST + OpenAI-`/v1` gateway + CLI; it
does **not** reimplement either server (vLLM lifecycle still goes through `serve.sh`
+ a socat relay in WSL).

- **Runs Windows-native** on `192.168.1.40` (`Alien-3070-TI`, Windows 11). Ollama is
  a Windows service; vLLM lives in **WSL2** (Ubuntu-24.04) and is driven over `wsl.exe`.
- **Primary lane** = RTX 3090 (24 GB). **Companion lane** (optional, off by default) =
  RTX 3070 Ti (8 GB). Lanes are independent and never evict each other.
- Canonical operational docs: the homelab wiki page
  `hosts/ollama-host/services/llmconfig` (and `hosts/ollama-host`). The repo is the
  source of truth for code; the wiki for how it's deployed on `.40`.

## The mental model (read once, it explains everything)

A **Lane** binds one inference-server *pair* (Ollama + vLLM) to **one GPU, matched by
UUID**. The core guarantee is the **eviction-wait gate** in `lane.py`: before loading
a target, the lane stops the other server + unloads all Ollama models on that card,
then **polls `nvidia-smi` (that GPU's UUID) until it's back to driver baseline**
(`vram_free_baseline_mb`) — only then does it load, so the model packs the whole card.
Every swap on a lane is serialized behind that lane's own `asyncio.Lock`.

- **Ollama** = one server, many models resident at once in principle; we keep exactly
  one by unloading the rest. Residency is controlled by `keep_alive` (`-1` = pin until
  swapped). Spill is detected via `/api/ps` `size_vram < size`.
- **vLLM** = one model per *process*. "Loading a different model" = restarting the
  templated systemd-user unit `vllm@<alias>` (which runs `serve.sh <alias>`). Status
  is read from the socat **relay** `/v1/models` (reports the currently-served name).

## Module map (`llmconfig/`)

- `config.py` — `Settings` (pydantic-settings, from `.env`/env) + `LaneConfig`;
  `settings.lanes()` builds the primary (always) and companion (if `companion_enabled`)
  lanes. `get_settings()` is `@lru_cache`d. **All box-specific values live here.**
- `main.py` — FastAPI app factory `create_app()`; all REST endpoints; lifespan that
  auto-loads lane defaults, starts the Monitor, and on shutdown stops the Monitor,
  releases the WSL keepalive, and closes pooled HTTP clients.
- `orchestrator.py` — `Orchestrator`: builds one `Lane` per `LaneConfig`, routes
  load/unload to the right lane, aggregates status (one `nvidia-smi` for all lanes via
  `query_all_gpus`), owns the **shared** `WslKeepalive` and `LaneDefaults`.
- `lane.py` — `Lane`: the per-GPU arbitration state machine (eviction-wait gate,
  `_load_ollama`/`_load_vllm`, `unload`, `_max_pack_reload`). **The heart of the app.**
- `lane_state.py` — `LaneDefaults`: persist each lane's startup-default model to
  `data/lane_defaults.yaml`.
- `backends/ollama.py` — `OllamaBackend`: REST client to the Ollama server + Windows
  service control. Pooled `httpx` client; `pull` uses a dedicated no-timeout client.
- `backends/vllm.py` — `VllmBackend`: relay `/v1/models` for status; `serve.sh` /
  `systemctl --user` over `wsl.exe` for lifecycle; `wait_ready`, `journal_tail`.
- `gpu.py` — nvidia-smi truth. `query_gpu(uuid)` (one card, w/ processes),
  `query_all_gpus()` (all cards — the multi-lane fast path),
  `sample_gpu_metrics()` (temp/power/util for the Monitor). Tries Windows nvidia-smi,
  falls back into WSL. `GpuInfo` carries `util_pct` (compute utilization, `None` when
  the driver reports `[N/A]`) and the `vram_pct` property (memory fraction) — see
  invariant 8 for why these must never be conflated.
- `nvapi.py` — pure-ctypes NVAPI wrapper for **hotspot + GDDR6X memory-junction**
  temps that nvidia-smi hides on consumer cards. Every failure path returns `None`.
- `monitor.py` — `Monitor`: background asyncio sampler → rolling in-memory deques +
  **best-effort SQLite** persistence (`data/monitor.db`) so history survives a restart.
  Backs the Monitor tab and `/api/monitor*`.
- `idle.py` — `IdleReaper`: background idle auto-unload policy. Reaps a lane after
  `idle_unload_after_min` of no activity (gateway request / load completion / Monitor
  util spike) so the card drops to P8, and releases the WSL keepalive when no lane
  serves vLLM. Participation is per lane (`LaneConfig.idle_unload_enabled`) — the
  companion is exempt by default (it idles in P8 anyway; `COMPANION_IDLE_UNLOAD_ENABLED`
  opts it in). Also `classify_usage()` — the shared free/idle/active classification
  behind `GET /api/usage` and the `usage` field on `/api/status` lanes.
- `registry.py` — `Registry`: the editable vLLM **alias catalog** (YAML at
  `data/vllm_models.yaml`, seeded from the package default). vLLM can't enumerate what
  it *could* serve, so this is that list.
- `schemas.py` — all pydantic models (`LoadRequest`, `StatusResponse`, `LaneStatus`,
  `VllmAliasEntry`, `Job`, …).
- `jobs.py` — `JobManager`: fire-and-forget async jobs with a streamed log (loads,
  pulls, downloads return a `Job`; the CLI/UI poll `/api/jobs/{id}`).
- `wsl.py` — `run_wsl()`, `WslKeepalive`, `user_systemctl`/`user_journalctl` helpers.
- `winsvc.py` — Windows service control (status/start/restart, elevation check).
- `proc.py` — `run_argv()` subprocess wrapper (`CmdResult`).
- `doctor.py` — read-only recon (`run_doctor`) that verifies every on-box assumption.
- `openai_gateway.py` — the OpenAI-compatible `/v1` gateway (auto-load on first request).
- `cli.py` — the `llmconfig` typer CLI (thin client over the REST API + `serve`).
- `web/` — static UI (`app.js`, `monitor.js`, `style.css`) + `templates/index.html`.
- `data/*.default.yaml` (in the package) — shipped registry seeds; `../data/*.yaml`
  (repo root) — the live, user-editable copies.

## Hard invariants — don't break these

1. **GPUs are identified by UUID, never by index.** Indices are unstable (the chassis
   3070 Ti flaps in/out of CUDA enumeration). `config.py` pins each lane by UUID and
   `gpu.py` matches on it. **vLLM/torch ordering is a trap:** nvidia-smi uses PCI_BUS_ID
   order (3090 = index 1) but vLLM 0.20.2's worker ignores `CUDA_DEVICE_ORDER` and uses
   CUDA FASTEST_FIRST (3090 = index 0). `serve.sh` resolves the index via the venv
   **torch** by UUID and hard-fails if absent — never silently index 0.
   **Ollama 0.30+ is a second trap:** its discovery also enumerates GPUs via **Vulkan**,
   which ignores `CUDA_VISIBLE_DEVICES` — an index pin silently lands models on the
   wrong card (found live 2026-07-08: companion's model on the 3090; ollama#16508).
   Both NSSM Ollama services therefore set `OLLAMA_VULKAN=0` + `GGML_VK_VISIBLE_DEVICES=-1`
   (CUDA-only discovery) and pin `CUDA_VISIBLE_DEVICES` by **UUID** (works once Vulkan
   is off). `deploy/install-companion.ps1` writes this; keep it if you touch the env.
2. **Lanes never touch each other's card.** vLLM stop is scoped to the lane's own
   systemd unit glob + its `serve.sh` path — **never a global `pkill -f venv/bin/vllm`**
   (that would cross-kill the other lane's vLLM when both GPUs serve). Keep it scoped.
3. **The eviction-wait gate is the contract.** Any new load path must evict + confirm
   VRAM freed (`_wait_vram_free`) before loading. Don't add a load that skips it.
   Likewise the idle reaper (`idle.py`) unloads **only** through `Lane.unload` (lane
   lock + eviction-wait) — never give it (or any future policy) a private unload path,
   and only release the WSL keepalive when no lane serves vLLM and no lane lock is held.
4. **Hold WSL open around vLLM.** WSL2 idle-shuts-down the whole distro ~seconds after
   the last `wsl.exe` exits, killing a just-loaded vLLM model *and* the relay — even
   with lingering. A vLLM load calls `keepalive.ensure()`; the app releases it on
   graceful shutdown. Don't remove the keepalive; don't forget to release it.
5. **Reach the relay at `127.0.0.1:11437`, never `localhost`.** Under WSL2
   localhost-forwarding, `localhost` triggers IPv4/IPv6 happy-eyeballs delays; a *down*
   relay blackholes the SYN (no RST) and hangs ~2.4 s — hence `vllm_probe_timeout_s`.
6. **Ollama context is baked into the Modelfile (`num_ctx`), not a load param.**
   `/api/load` and the `/v1` gateway set only `keep_alive` and optional `num_gpu`. To
   change context, bake a new tag (`ollama create <m>-64k -f Modelfile`) — do **not**
   add a context arg to the load path.
7. **Write endpoints are gated by `X-API-Key` only when `LLMCONFIG_API_KEY` is set**
   (`require_key` dependency). Read/inference endpoints are open (LAN perimeter). Keep
   new mutating endpoints on the `write` dependency list.
8. **`utilization_pct` means compute load, never VRAM occupancy.** `/api/status`'s
   `gpu.utilization_pct` is nvidia-smi `utilization.gpu` (nullable); the memory
   fraction lives in `vram_pct` (and `loaded.gpu_vram_pct`). Until 9e55316 the field
   carried the VRAM fraction, so external idle gates (LocalLLM_Code_Analysis's
   `Wait-GpuIdle`) saw a resident model as ~86% "busy" forever and deadlocked their
   runs. Off-box consumers key off this field — don't swap the semantics back.
9. **Adding a vLLM model needs a `serve.sh` case, not just a registry row.** The
   registry's `launch_args` / `managed_by: registry` fields are currently **unwired** —
   `_load_vllm` always launches via `vllm@<alias>` → `serve.sh <alias>`, whose hardcoded
   `case` sets the args. A user-added model = add a `case` to `deploy/serve.sh` **and**
   add the alias row. If you wire up `managed_by: registry`, update `lane.py` + doctor.

## Build / run / test

```powershell
# from the repo root, on the box (or any Windows/WSL host with the GPUs)
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\llmconfig doctor --local     # verify on-box assumptions BEFORE trusting swaps
.\.venv\Scripts\llmconfig serve              # uvicorn on :11430 (UI + REST + /v1)
pytest                                        # unit tests (no GPU needed; httpx mocked via respx)
```

- **Tests** live in `tests/`, `asyncio_mode = auto` (see `pyproject.toml`). They mock
  Ollama/vLLM HTTP with `respx` and stub `nvidia-smi`/`wsl.exe`; they do **not** need a
  real GPU. Add tests alongside the module you touch (`test_lane_companion.py`,
  `test_orchestrator.py`, `test_gpu.py`, `test_monitor.py`, `test_openai_gateway.py`,
  `test_registry.py`, `test_wsl.py`).
- The CLI is a thin client — point it anywhere with `--url` / `$LLMCONFIG_URL`
  (e.g. `http://192.168.1.40:11430` over Tailscale).
- `doctor` runs read-only; run it after any change to the WSL/serve.sh/unit plumbing.

## Deploy-time gotchas (see `deploy/README-deploy.md` for the full runbook)

- **Run the app elevated** (or as a LocalSystem service) — it must `Restart-Service`
  Ollama; a plain user shell hits "access denied". Always-on = Scheduled Task `LLMConfig`
  at logon as `folar`, RunLevel Highest (a LocalSystem service can't drive `wsl.exe -u
  folar`, which needs the user session — that's why it's a task, not NSSM).
- **Restart cleanly.** `Stop-ScheduledTask LLMConfig` can leave the uvicorn child
  holding `:11430`; kill it before `Start-ScheduledTask` or the new instance wedges.
- **Never run the Ollama tray app (`ollama app.exe`).** Its auto-updater can't stop the
  NSSM service, corrupts the in-place update, and silently drops Ollama to **CPU-only**
  (`library=cpu`). Update via `deploy/update-ollama.ps1` (a weekly task automates it and
  verifies the CUDA runner offloads to GPU afterward).
- **Cache-busting:** `main.py` tags `/static/*` URLs with the newest asset mtime, so a
  redeploy isn't masked by a stale `style.css`/`app.js`. Keep that if you touch the UI.
- **Ports:** LLMConfig `11430`; Ollama `11434` (companion `11435`); vLLM relay `11437`
  (companion `11438`). All LAN-only, no auth by default.

## Conventions

- Windows-first but must degrade off-box: nvidia-smi/NVAPI/wsl failures return
  empty/None and the feature simply goes quiet (see `gpu.py`, `nvapi.py`, `monitor.py`).
  Preserve that — a missing tool must never crash a request or the sampler loop.
- Long operations (load/unload/pull/download) return a `Job` and stream a log; don't
  block a request thread on them.
- Force UTF-8 on Windows console streams (the CLI does this in `main()`) — report
  glyphs (`— … → ●`) become mojibake under cp1252 otherwise.
- Match the surrounding style: dense module docstrings that explain *why*, type hints,
  `from __future__ import annotations`. New env-configurable values go in `config.py`
  **and** `.env.example`.

## Don't

- Don't pin a GPU by index anywhere, or add a global `pkill` for vLLM.
- Don't add a load path that skips the eviction-wait gate or the WSL keepalive.
- Don't add an Ollama context-length load param (bake a tagged Modelfile instead).
- Don't commit `.env`, `data/*.yaml` live copies, or `data/monitor.db*` (gitignored).
- Don't rewrite `serve.sh` behavior from Python — vLLM lifecycle stays in serve.sh + units.
