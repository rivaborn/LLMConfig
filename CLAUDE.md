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
  The companion is **Ollama-only** (`COMPANION_VLLM_ENABLED=false`): its
  `serve-companion.sh` was specified in `deploy/vllm-companion@.service` but never
  written, so `LaneConfig.vllm_enabled=False` makes that explicit — doctor reports
  configuration instead of a permanent FAIL, the catalogs (`/api/models`,
  `/v1/models`) stop advertising aliases that cannot launch, both resolvers skip
  the lane's vLLM registry, and `_load_vllm` refuses BEFORE evicting the lane's
  working Ollama model. Build the script and flip the flag to re-enable.
- Canonical operational docs: the homelab wiki page
  `hosts/ollama-host/services/llmconfig` (and `hosts/ollama-host`). The repo is the
  source of truth for code; the wiki for how it's deployed on `.40`.

## The mental model (read once, it explains everything)

There are three kinds of **unit**. The first two are UI-visible and treated
identically (Home tab = one box per unit; one control tab per unit; Monitor
covers all of them):

- a **`Lane`** — one local GPU arbitrated Ollama-XOR-vLLM (3090 / 3070 Ti);
- a **`SparkUnit`** — one remote DGX Spark node driven by `sparkrun`;
- a **`SparkGroup`** — a SET of Spark nodes serving ONE tensor-parallel model
  over the 200G fabric. **Synthetic**: it lives in `orch.units` (so placement,
  leases, and the /v1 gateway treat it like any unit) but `settings.units()`
  never emits it — no tab, no Home card — and it is filtered out of
  `/api/status` lanes. The members' own cards carry the residency
  (`LoadedModel.group`, the "×2 · spark1+spark2" badge). Groups exist only when
  **`SPARK_FABRIC_ENABLED`** is on (default off — the switch is on order); off,
  `orch.units` is byte-for-byte pre-multi-node and the Cluster tab is a planner.

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
- `spark_group.py` — `SparkGroup`: a **set of Spark nodes** serving one
  tensor-parallel model (see the mental model). Load = ordered bounded member
  locks → whole-node eviction per member → ONE `sparkrun run --hosts h1,h2 --tp K`
  on the head → wait-ready on the head port → claim + placement record.
- `group_state.py` — `GroupState` (live claims, all-sync — members read it in
  `status()`/`_admit`) + `GroupPlacements` (persisted (model, node-set) history,
  `data/spark_group_state.yaml` — feeds startup group re-instantiation and
  auto-placement).
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
- `fsio.py` — `atomic_write_text()`: every YAML `save()` goes through it (temp +
  `os.replace`); a mid-write power cut must never torch a state file (invariant
  17's reboots are a live path). Route any new persisted-file write through it.
- `doctor.py` — read-only recon (`run_doctor`) that verifies every on-box assumption.
- `openai_gateway.py` — the OpenAI-compatible `/v1` gateway (auto-load on first request;
  chat + completions + embeddings/rerank/score; auto-placement wiring in `_choose`).
- `cookbook.py` — named fleet states: snapshot residency, apply-exactly (one meta-job,
  units parallel, unload-extras-then-load-missing, `needs_empty_node` forces a full
  rebuild), mark-default (syncs `LaneDefaults` incl. `[]` tombstones).
- `load_times.py` — measured launch durations (units record success-only, launch-span
  only; Sparks share one key per alias) + per-unit consecutive-failure counters
  (separate `failures:` section — never mixed into the medians). Behind
  `/api/load-times`, `/api/load-times/{model}`, the UI estimates, and placement's
  proven-load gate + blocklist.
- `placement.py` — auto-placement: pure `rank()` over per-unit `CandidateFacts` +
  `Placer` (single-flight status sweep, TTL'd Ollama tag cache — its only I/O —
  sync lease/registry reads). Advisory by design — see invariant 15. Every
  `place()` records into a ~50-entry decision ring buffer (`/api/placement/
  decisions`, consecutive routine repeats deduped) — the debugging surface for
  "why did that land there", since the always-on task's console isn't captured.
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
   The effective window is `baked num_ctx, else OLLAMA_CONTEXT_LENGTH` (**32768** in both
   NSSM services' `AppEnvironmentExtra`); an *absent* variable silently falls back to
   Ollama's built-in 4096, which is what once made the companion relay serve 4 k.
   **When reading a window, `/api/ps` `context_length` is the runtime truth and
   `/api/show` `parameters.num_ctx` is the configured bake — never
   `/api/show` `model_info.*context_length*`,** which is the *architectural* maximum and
   is wildly larger (`qwen3.6:27B` reports 262144, `devstral-small-2` 393216 — both
   actually serve 32768). Sizing anything off that field produces limits up to 8× the real
   window that look correct and 400 on a long prompt. `schemas.py` and
   `backends/ollama.py` already say this; the trap is re-derived from scratch every time.
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
   on `stop`** (with neither it exits *"Must specify TARGET or --all."*).
   **A targeted stop must use the sparkrun JOB ID, never the recipe name** —
   `sparkrun stop <recipe> --hosts` prints *"Workload stopped"* with rc=0 and stops
   NOTHING (live, 2026-07-26: the embedder survived three "successful" stops), and
   `stop` also swallows SSH failures into rc=0, so the exit code can never be
   trusted; the slot re-probe after unload is the real check.
   `SparkBackend.stop(recipe=…)` therefore resolves the recipe to this host's job
   id via `sparkrun status` and stops by id (`SPARK_STOP_JOB_CMD`); `SPARK_STOP_CMD`'s
   `--all` sweeps the node for "free the whole unit". `SPARK_STOP_ONE_CMD` is a
   documented tombstone.
   **A catalog entry's `extra_args` (→ `{extra}`) is load-bearing, including
   `-o env.KEY=VAL`.** Two things ride there and neither is cosmetic: the
   `--max-model-len` cap (upstream recipes are sized for TWO nodes, so a one-node
   launch must cap down — see `ContextUpdate.md` for the per-recipe table) and recipe
   **env overrides**. `qwen35-122b-int4` carries
   `["-o", "env.VLLM_MARLIN_USE_ATOMIC_ADD=0"]` because the `@eugr` recipe ships that
   env as `1` and on GB10 the marlin split-K path dies during CUDA-graph capture
   (capture 23 of 51, *"illegal instruction"*). That model was blamed on context for
   days — it booted at 94.8 % pool and died on its FIRST token, so 262144 was cut to
   65536 twice before the env was found to be the real cause (weights 62.87 GiB, KV
   31.2 GiB free the whole time; it now serves the full 262144). **Before shrinking a
   Spark model's window, check whether a recipe env default is killing it.**
   Recipe names are **namespaced** (`@eugr/…`, `@official/…`)
   — find them with `sparkrun search`.
   `tests/test_spark.py::test_launch_command_matches_verified_sparkrun_flags` and
   `::test_targeted_stop_resolves_the_job_id_on_this_host` pin this.
11. **`LeaseManager`'s query/mutation methods must stay synchronous.** And the
   unit swap-lock acquire must stay a BARE `await lock.acquire()` on the
   uncontended path (`_acquire_swap_lock` in both unit kinds): `asyncio.wait_for`
   wraps the acquire in a task and always yields even on a free lock, re-opening
   the reaper/sweeper check-then-act window this invariant closes. Also:
   `claim()` contends with EVERY overlapping live lease (a whole-unit claim
   overlaps them all) — never reduce it back to first-match. `idle.py`'s
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
   **The proven-load gate** (`PLACEMENT_REQUIRE_PROVEN`, 2026-07-28) and the
   consecutive-failure blocklist are further *ranking predicates*, not gates: a
   fresh load is only CHOSEN on a unit where the model has launched successfully
   before (a `LoadTimes` sample, or current residency), and a unit with
   `PLACEMENT_FAIL_BLOCK_AFTER` consecutive launch failures is skipped until the
   cooldown lapses. The tiers that don't choose are exempt on purpose — the
   sole-candidate pin and the resident tier — which is how a first-ever load
   gets seeded (pin, or an explicit lane). Mind the key asymmetry: success
   samples are FLEET-WIDE for Sparks (`spark:{alias}` — identical GB10s, and
   residency on any spark proves all four) while failure counters are PER-UNIT
   for everyone (`fail_key` — launch failures are usually node-state-dependent,
   so spark1 failing must not block spark2). "Proven" means the LAUNCH
   succeeds, not that inference is stable.
   **Workload tiering** (2026-07-29) is a third predicate family, again a
   preference and never a gate: the 3090 is the speed tier, the Sparks the
   capacity tier. `classify_workload` (pure, in placement.py) reads the request
   body — interactive prefers the GPU lane *after* idle-first (a latency
   request never queues behind an active model); batch prefers a Spark
   *before* idleness (an active Spark absorbs one more request into its batch;
   bulk work must not occupy the speed tier). `X-LLM-Workload` overrides. No
   workload (REST paths, kill switch) = the neutral ordering byte-for-byte —
   don't add a call site that fabricates a Workload it didn't classify.

14. **A Spark load is admitted by summed `mem_fraction`, and there is nothing
   else.** A lane can watch `nvidia-smi` drain before it loads; a Spark has no
   equivalent — memory comes back only when a container stops, and vLLM
   preallocates, so a resident model cannot be shrunk to make room. The declared
   budgets in the catalog ARE the contract: `SparkUnit._admit` refuses a load when
   the resident budgets plus the new one exceed `SPARK_MEM_HEADROOM` (0.95), and
   refuses co-residency with any model whose `mem_fraction` is unset (0.0 = a
   whole-node claim). Don't add a Spark load path that skips `_admit`, and keep new
   catalog entries budgeted or they silently monopolise a node. The UI's gray-out
   (`SparkModel.addable`) is computed in `SparkBackend.list_models` beside this same
   arithmetic (`declared_budgets`) — never re-implement it client-side.
   **`needs_empty_node` is the second half of admission and is NOT arithmetic.**
   Some recipes are lethal to co-residents at LAUNCH regardless of budgets: the
   reranker's fastsafetensors path kills them at driver level, gemma's
   quantize-at-load transient (~74 GB) trips Ray's 95% ceiling. `SparkUnit._load`
   refuses such a load while anything is resident (after victims/reload-stop/tp>1
   sweep, before `_admit`); `force=true` overrides with a loud job-log warning.
   It lives in `_load` because EVERY path funnels there — the cookbook's apply
   frees the node first, but the boot autoload did not, and on 2026-07-28 it
   launched the reranker beside a resident gemma and destroyed it (0.35 + 0.40
   fits, which is exactly why budgets can't catch this). `placement.py` mirrors it
   by treating `needs_empty_node` as a `whole_node` claim so ranking stops
   proposing populated nodes.
17. **Nothing may wait on WSL without a bound, and the boot restore is gated.**
    Windows Update auto-restarted the box at 00:29 on 2026-07-28 (its permitted
    window is 00:00-06:00; no AU policy is set, so this recurs). The box was back in
    60 s; the lab was degraded for six hours. Three rules came out of it:
    - **`run_argv` must always return.** It ran `proc.kill()` then a bare
      `await proc.wait()`. `kill()` is a *request* — a process wedged in an
      uninterruptible kernel call ignores it, so `wait()` never returned and the
      module's own "a hang becomes rc 124" contract broke silently. Every external
      command funnels through here, so that one line caused a 5 h lock hold, an
      orphaned `wsl.exe` pile-up and a stalled Monitor simultaneously. The reap is
      bounded (`REAP_TIMEOUT_S`); a survivor is abandoned and still reports 124.
    - **Probe WSL by EXEC, never by `--status`.** Throughout the incident
      `wsl --status` answered normally while every `wsl -u folar` hung forever.
      Only the exec path (`wsl.probe`) tells the truth. `main.py` gates the boot
      `autoload_defaults()` on `wsl.wait_ready()` — in the background, so uvicorn
      still binds immediately, and non-fatally, so a box with no `wsl.exe` still
      serves (rc 127).
    - **Unit locks are acquired with a timeout, on load AND unload.** An unbounded
      `async with self._lock` let one wedged holder stack 29 jobs while the unit
      read as "busy" rather than broken. Unload is included on purpose: it is the
      natural way out of a wedge, and during the incident the one call that should
      have cleared it was queued behind it.

    `WslRecovery` escalates a wedged distro in the order that actually worked: kill
    orphans → attempt `--shutdown` (**it timed out twice and never reaped the
    utility VM — attempted, never trusted**) → restart `WslService`, the only
    effective step (~16 stop-poll cycles, hence the timeout arg on
    `winsvc.restart_service`).

18. **A multi-node (SparkGroup) load holds EVERY member's lock, acquired in
    `orch.units` order, bounded, all-or-nothing** — the one global order is what
    keeps overlapping group loads (spark1+spark2 vs spark2+spark3) queueing
    instead of deadlocking; a failed acquire releases the prefix. Under those
    locks the load re-validates member claims and eviction victims (the
    `placement_conflict:` protocol), frees each member WHOLE (v1: no cross-node
    co-residency arithmetic), then launches ONCE via the head with
    `--hosts <all members>`. Corollaries: **members refuse their own loads and
    unloads while group-claimed** (`_admit`/`unload` — a `stop --all` on one
    rank wedges the others; teardown stops the sparkrun JOB ID, which is
    cluster-wide, then re-probes); **multi-node recipes live ONLY in the
    cluster catalog** (`data/spark_cluster_models.yaml`), never per-node — the
    disjointness is what makes a model resolve on groups XOR single units so
    `rank()` never arbitrates between a group and its own members; and the
    whole feature is inert while `SPARK_FABRIC_ENABLED` is false (no group
    units exist — don't create one anywhere else). `GroupState` reads are sync
    on purpose (invariant 11 applies to it too).


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
- **Changing a served context is a cross-repo change.** `rivaborn/opencode-config`'s
  `opencode.json` pins `limit.context + limit.output == served ceiling` per model, because
  opencode uses `output` as a fixed `max_tokens` and caps the prompt at `context` with no
  reservation — so a ceiling change silently makes it either wasteful (short) or 400-prone
  (long). Any edit to a `serve.sh` `--max-model-len`, a Spark `extra_args` cap, a baked
  Ollama `num_ctx`, or `OLLAMA_CONTEXT_LENGTH` obliges a re-sync there. `ContextUpdate.md`
  holds the current per-tier tables; audit with
  `curl -s :11430/v1/models | jq -r '.data[].id'` (no lane header lists the whole fleet).
  Model **ids** matter as much as limits: they match by exact string, so a renamed
  `served_name` is a silent 404 at `/model` time, not a config error.
- Pooling/OCR models (`qwen3-embed`, `surya-ocr-2`, `qwen3-vl-embedding-8b`,
  `qwen3-vl-reranker-8b`) are deliberately **absent** from `opencode.json` — they serve
  `/v1/embeddings`, `/v1/rerank` and `/v1/score`, not chat. `/v1/models` lists them, so
  that id diff is expected; don't "fix" it.

## Don't

- Don't pin a GPU by index anywhere, or add a global `pkill` for vLLM.
- Don't add a load path that skips the eviction-wait gate or the WSL keepalive.
- Don't add an Ollama context-length load param (bake a tagged Modelfile instead).
- Don't commit `.env`, `data/*.yaml` live copies, or `data/monitor.db*` (gitignored).
- Don't rewrite `serve.sh` behavior from Python — vLLM lifecycle stays in serve.sh + units.
