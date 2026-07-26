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

There are two kinds of **unit**, and the UI/API treat them identically (Home tab =
one box per unit; one control tab per unit; Monitor covers all of them):

- a **`Lane`** — one local GPU arbitrated Ollama-XOR-vLLM (3090 / 3070 Ti);
- a **`SparkUnit`** — one remote DGX Spark node driven by `sparkrun`.

`settings.units()` = `lanes()` + `sparks()`; the orchestrator keys everything off
`self.units`, while `self.lanes` stays GPU-only (the WSL keepalive and the vLLM
release logic are meaningless for a remote node). Sparks are **off by default**
(`SPARK_ENABLED`).

**The two kinds differ in how many models they hold, and that is not a style
choice.** A lane holds exactly ONE, enforced by the eviction-wait gate below,
because a 24 GB card that overcommits spills into system RAM. A Spark has ~121 GB
of unified memory and holds up to `SPARK_MAX_MODELS` (default 4) — one per slot
port, `api_port … api_port+N-1` — because there is nothing to spill into and an
embedder occupying a whole node at 17 % utilisation is pure waste. So on a Spark:

- **Ports are discovered, never persisted.** `status()` probes all N slots
  concurrently over HTTP and builds the port→model map from what answers, which is
  restart-safe and keeps invariant 9 (status never awaits SSH).
- **`loaded` is the primary occupant; `loaded_models` is the truth.** The scalar
  stays for back-compat and always equals `loaded_models[0]`; a lane populates the
  list with its single model so "empty iff `loaded is None`" holds for both kinds.
- **Admission is by summed `mem_fraction`** — see invariant 14.
- **Policy is per model**: per-model activity clocks (`touch(model=…)`), one victim
  per reaper tick, leases that may name a model, and `unload(model=…)` for a
  targeted stop. Any model name entering that machinery is folded onto the catalog
  alias by `SparkUnit.canonical_model` — the gateway, a load and residency name the
  same model three different ways.
- **`tp > 1` claims the whole node** and must not be co-scheduled.

The rest of this section describes the GPU-lane half.

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
- `spark_unit.py` — `SparkUnit`: one **remote DGX Spark node** as a unit. Same
  duck-typed surface as `Lane` (`status`/`load`/`unload`/`touch`/`aclose`, own
  `asyncio.Lock`, Job pattern) but no eviction gate and no local card. It holds
  several models at once, one per slot port, so a load is admit → run → wait-ready
  (only a *reload of the same alias* stops anything first, and then only its own
  slot). `unload(model=…)` stops one; without a model it frees the node.
- `backends/spark.py` — `SparkBackend`: three transports, each the most reliable
  source for its job — **status over HTTP** (the node's `/v1/models`), **lifecycle
  via the `sparkrun` CLI** through `run_wsl`, **telemetry over SSH** (`nvidia-smi`,
  parsed by `gpu.py`).
- `lane_state.py` — `LaneDefaults`: persist each unit's startup-default model**s** to
  `data/lane_defaults.yaml` (a list per unit — a Spark starts several; the older
  scalar shape is read and migrated on the next write).
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
- `leases.py` — `LeaseManager` + `LeaseSweeper`: **resource sharing between callers**.
  A lease records who holds a unit, whether their work may be interrupted, and a
  renewable TTL; a displaced holder learns via `state=revoked` on poll **and** an
  in-band 409 on its next `/v1` call. Every method is **sync on purpose** — see
  invariant 11.
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
- `openai_gateway.py` — the OpenAI-compatible `/v1` gateway (auto-load on first request;
  chat + completions + embeddings/rerank/score; auto-placement wiring in `_choose`).
- `placement.py` — auto-placement: pure `rank()` over per-unit `CandidateFacts` +
  `Placer` (single-flight status sweep, sync lease/registry reads). Advisory by design
  — see invariant 15.
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
   Likewise any policy that unloads (the idle reaper, the lease sweeper, placement
   eviction) goes **only** through `Unit.unload` or the unit's OWN load path under its
   own lock (`SparkUnit._load` stops a reload target, a tp>1 sweep, and placement
   victims there — each re-validated under the lock) — never a private unload path.
   Only release the WSL keepalive when no lane serves vLLM and no lane lock is held.
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
9. **A Spark unit must never block `/api/status` on SSH.** The UI polls status every
   2.5 s; a node's `nvidia-smi` is an SSH round-trip and a powered-off node costs the
   full connect timeout. `SparkUnit.status()` therefore awaits **only** the fast HTTP
   probe, serves the last telemetry sample, and refreshes it in a background task
   (`_refresh_gpu_soon`); a repeated-failure breaker also backs off the HTTP probe.
   Keep both — without them one dead Spark adds seconds to every status call for
   every client. `tests/test_spark.py::test_status_never_awaits_ssh` guards this.
10. **`sparkrun` command templates live in `Settings`, not in code.** Its flags shift
   between releases, so `SPARK_RUN_CMD` / `SPARK_STOP_CMD` / `SPARK_SSH_CMD` are
   `.env`-tunable. Fix a flag mismatch there, not by hardcoding a new argv.
   Verified against **sparkrun 0.2.40** (2026-07-24) by `--dry-run` on the live
   cluster; three flags are load-bearing and each fails differently if dropped:
   **`--cluster` alongside `--hosts`** (the cluster carries the SSH user — with
   `--hosts` alone sparkrun uses the local WSL user and every node returns
   *Permission denied*), **`--no-follow`** (otherwise `run` tails container logs
   and never returns, so the load hangs to its timeout), and **a TARGET or `--all`
   on `stop`** (with neither it exits *"Must specify TARGET or --all."*). There are
   two stop templates and the difference matters: `SPARK_STOP_ONE_CMD` names a
   recipe and leaves co-residents running — it is what per-model unload and the
   idle reaper use — while `SPARK_STOP_CMD`'s `--all` sweeps the node and is only
   for "free the whole unit". Recipe names are **namespaced** (`@eugr/…`,
   `@official/…`) — find them with `sparkrun search`.
   `tests/test_spark.py::test_launch_command_matches_verified_sparkrun_flags` pins this.
11. **`LeaseManager`'s query/mutation methods must stay synchronous.** `idle.py`'s
   final guard and `LeaseSweeper._free_unit` both rely on there being **no await**
   between the last check and `Unit.unload()` — an uncontended `asyncio.Lock`
   acquires without yielding, so that pair is atomic. Make `active_for` (or any of
   its siblings) a coroutine and a competing load can interleave and have its
   freshly loaded model unloaded out from under it.
   `tests/test_leases.py::test_query_methods_are_sync` pins this.
16. **`X-LLM-Hold` claims a PREEMPTIBLE lease, deliberately.** A static-config client
   (opencode) cannot send a lease id — none exists until claimed — so the header asks
   the gateway to claim/renew one for it. It is preemptible because the goal is
   "don't displace my model automatically" (idle reaper, placement eviction), NOT
   "refuse everyone else" — a non-preemptible auto-hold would 409 every other client
   the moment opencode touched a shared model. It never preempts an existing holder,
   never raises into the request, and lapses `AUTO_HOLD_TTL_S` after the last request.
   Non-preemptible exclusivity stays a deliberate, manual `/api/leases` claim.

12. **A lease is additive on the API, never a fourth `usage` value.** `LaneUsage`
   stays `free|idle|active` because off-box consumers switch on it (see invariant 8);
   the claim rides alongside as `lanes[].lease`. Leases are also **advisory** —
   direct-to-Ollama clients are ungated — and a non-preemptible lease is a *forward*
   guarantee only: work already running keeps the unit (there is no job cancellation).
   A non-preemptible lease gates `/v1` **and** `POST /api/load`/`/api/unload` (the
   holder authorizes itself via `X-LLM-Lease`) — guarding only `/v1` would let anyone
   swap the model over the held unit through the REST path.
13. **Adding a vLLM model needs a `serve.sh` case, not just a registry row.** The
   registry's `launch_args` / `managed_by: registry` fields are currently **unwired** —
   `_load_vllm` always launches via `vllm@<alias>` → `serve.sh <alias>`, whose hardcoded
   `case` sets the args. A user-added model = add a `case` to `deploy/serve.sh` **and**
   add the alias row. If you wire up `managed_by: registry`, update `lane.py` + doctor.
15. **Auto-placement is advisory; the gates are elsewhere.** A `/v1` request without
   `X-LLM-Lane` (or with `X-LLM-Lane: auto`) lets `placement.py` pick the unit:
   resident first, then free capacity, then eviction of idle+unleased models — but
   the ranking runs on a snapshot, so nothing may TRUST it. Admission (`_admit`),
   the unit lock, the lease gate, and the under-lock re-validation of eviction
   victims (`_evict_victim`, refusing with `placement_conflict:`) are the real
   gates; the gateway answers a conflict with ONE re-place, never a loop. A model
   resolving on exactly one unit bypasses ranking entirely (sole-candidate pin), so
   a single-unit deployment behaves exactly as an explicit header would.
   `AUTO_PLACE_ENABLED=false` restores the implicit-primary default byte-for-byte.
   The id `auto` is reserved and can never name a unit.

14. **A Spark load is admitted by summed `mem_fraction`, and there is nothing
   else.** A lane can watch `nvidia-smi` drain before it loads; a Spark has no
   equivalent — memory comes back only when a container stops, and vLLM
   preallocates, so a resident model cannot be shrunk to make room. The declared
   budgets in the catalog ARE the contract: `SparkUnit._admit` refuses a load when
   the resident budgets plus the new one exceed `SPARK_MEM_HEADROOM` (0.95), and
   refuses co-residency with any model whose `mem_fraction` is unset (0.0 = a
   whole-node claim). Don't add a Spark load path that skips `_admit`, and keep new
   catalog entries budgeted or they silently monopolise a node.


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
