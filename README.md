# LLMConfig

A control plane for every **LLM unit** in the lab. It answers *what can run, what is
running, and where* — and it makes switching models a single click or one API call,
with the guarantee that a model gets the hardware to itself.

Two kinds of unit sit behind one Web UI + REST API + OpenAI-`/v1` gateway + CLI:

- **GPU lanes** — a local card arbitrated **Ollama ⇄ vLLM**, with an eviction-wait gate
  that guarantees the chosen model is the *only* thing in VRAM so it never spills to
  system RAM prematurely. The **RTX 3090 (24 GB)** and an optional **RTX 3070 Ti (8 GB)**.
- **DGX Spark nodes** — remote **GB10** boxes driven by [`sparkrun`](https://sparkrun.dev/).
  The node *is* the unit: it runs one workload, so a swap is stop → run → wait-ready.

Runs Windows-native on `192.168.1.40` (`Alien-3070-TI`, Windows 11), reaching vLLM
inside **WSL2** and the Sparks over the LAN.

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

- **One dashboard for every unit.** A Home tab with a box per unit — loaded model, the
  real HF repo behind it, served context window, memory bar — plus a quick-switch
  dropdown. Then one control tab per unit, and a Monitor covering all of them.
- **Pick + load** a model on any unit via Web UI / REST / CLI, as a streamed job.
- **Guaranteed packing (GPU lanes):** evict → confirm-freed via `nvidia-smi` → load.
  Reports on-GPU vs on-CPU bytes and flags *premature* spill; `--max-pack` pushes
  `num_gpu` to fill VRAM first.
- **Leases** — a real claim on a unit: who holds it, whether their work may be
  interrupted, how long they need it, and an in-band answer when they get displaced.
- **OpenAI `/v1` gateway** with **auto-load on first request**, so a client's `/model`
  picker switches models with no manual swap. `X-LLM-Lane` pins a unit; a request
  **without** the header is **auto-placed** — LLMConfig picks the unit (resident first,
  then free capacity, then evicting an idle unleased model) and answers with
  `X-LLM-Unit` saying where it ran.
- **Served context window** surfaced per unit — the window a prompt budget must
  actually respect, not the model's architectural maximum.
- **Independent units.** The 3090 can serve a big vLLM model while the 3070 Ti serves a
  small one and all four Sparks serve their own — no cross-unit eviction.
- **Monitor:** live thermals (core + hotspot + GDDR6X junction), power, memory and the
  Ollama GPU/CPU split, with rolling history persisted across restarts.
- **Idle auto-unload** so an unused card drops to P8 (~115 W → ~30 W measured).
- **Model management:** pull/delete Ollama models, edit the vLLM and Spark catalogs,
  trigger HuggingFace downloads — all as streamed jobs.
- **Cookbook:** save the current fleet arrangement (which models run where) under a
  name; applying a state loads exactly those models — and only those — across every
  unit as one streamed meta-job, with lease-aware unloads and the load-order rules
  (`needs_empty_node`) honoured. One state can be the startup default.
- **Load-time estimates:** every real launch is timed (median of the last 5); the UI
  shows sizes + "≈2 min load" on hover, and grays models that won't fit a node —
  computed server-side beside the admission arithmetic so the two can't drift.
- **`doctor`:** read-only recon that verifies every on-box assumption before you trust
  a swap.

## Topology

|            | Ollama              | vLLM                                   | DGX Spark                          |
| ---------- | ------------------- | -------------------------------------- | ---------------------------------- |
| Runs in    | Windows 11 native   | WSL2 Ubuntu 24.04                      | the remote GB10 node (Docker)      |
| Reach      | `127.0.0.1:11434`   | `127.0.0.1:11437` (socat relay)        | `http://192.168.1.5x:8000`         |
| Model swap | REST `keep_alive`   | one model/process — `serve.sh <alias>` | per-model `sparkrun stop/run`, slot ports 8000-8003 |
| State via  | `/api/ps`           | relay `/v1/models`                     | the node's own `/v1/models`        |
| Catalog    | `ollama list` tags  | `data/vllm_models.yaml`                | `data/spark_models_<unit>.yaml`    |

The control app listens on **`:11430`** (UI + REST + `/v1`). Everything is **LAN-only**
with no auth unless you set `LLMCONFIG_API_KEY`.

### The units

| Unit id                 | Hardware                     | Arbitration                                 | Telemetry                        |
| ----------------------- | ---------------------------- | ------------------------------------------- | -------------------------------- |
| `primary`               | RTX 3090, 24 GB              | eviction-wait gate, Ollama **XOR** vLLM     | local `nvidia-smi` by UUID       |
| `companion`             | RTX 3070 Ti, 8 GB (opt-in)   | same, fully independent of `primary`        | local `nvidia-smi` by UUID       |
| `spark1`…`spark4`       | DGX Spark GB10, 128 GB       | up to 4 co-resident models, one per port; admission by summed `mem_fraction` | remote `nvidia-smi` over SSH     |

`settings.units()` = `lanes()` + `sparks()`. Sparks are **off by default**
(`SPARK_ENABLED=true` to enable). Both kinds satisfy the same duck-typed contract, so
the UI, CLI, gateway and idle reaper treat them through one code path.

> **GB10 has no private VRAM.** The GPU shares the host's unified LPDDR5X pool, so
> `nvidia-smi` returns `[N/A]` for *every* memory field on a Spark — occupancy is read
> from the node's `/proc/meminfo` instead. Without that fallback a fully-loaded node
> reports 0 %.

## How it works

```
                Web UI  ·  CLI  ·  REST  ·  OpenAI /v1
                                │
                    FastAPI (Windows-native, :11430)
                                │
                 Orchestrator  ·  LeaseManager  ·  IdleReaper
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
  Lane: primary          Lane: companion            SparkUnit ×4
  (RTX 3090)             (RTX 3070 Ti)              (GB10 nodes)
  ┌─────┴─────┐          ┌─────┴─────┐              ┌────┴────┐
Ollama     vLLM        Ollama     vLLM          status    lifecycle
:11434    relay :11437 :11435   relay :11438    HTTP      sparkrun
Win svc   wsl.exe →    NSSM svc  wsl.exe →      :8000     via wsl.exe
          serve.sh               serve-comp.sh            └ telemetry:
  └─────┬─────┘          └─────┬─────┘                      ssh nvidia-smi
 nvidia-smi by UUID     nvidia-smi by UUID                  + /proc/meminfo
 eviction-wait gate     eviction-wait gate                 (no gate needed)
```

- A **Lane** binds one Ollama+vLLM pair to **one GPU, matched by UUID** (indices are
  unstable). Each lane arbitrates independently behind its own lock; lanes never evict
  each other.
- The **eviction-wait gate** is the core guarantee: evict everything else on the card,
  poll `nvidia-smi` until it's back to driver baseline, *then* load — 100 % VRAM before
  any spill.
- vLLM serves **one model per process**; a swap restarts the `vllm@<alias>` systemd
  unit (which runs `serve.sh`). vLLM status is read from the socat relay's `/v1/models`.
- A **SparkUnit** needs no gate — the node runs one workload, so a swap is
  stop → run → wait-ready. It reads status over HTTP, drives lifecycle through the
  `sparkrun` CLI, and takes telemetry over SSH; **`status()` never awaits SSH**, because
  the UI polls it every 2.5 s and one powered-off node would otherwise add seconds to
  every call for every client.
- The Orchestrator holds a **shared WSL keepalive** (`wsl.exe … sleep infinity`) so the
  distro — and any loaded vLLM model + relay — survives WSL2's idle-shutdown. Sparks
  need no keepalive: their workload lives on the node.

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
| `GET /api/usage?lane=`                  | Compact tri-state per lane: `free` / `idle` / `active` + model|
| `GET /api/lanes`                        | Configured lanes (id, name, enabled, current default)         |
| `GET /api/models?lane=`                 | That lane's Ollama tags + vLLM alias catalog (loaded flagged) |
| `GET /api/gpu?lane=`                    | Parsed `nvidia-smi` for that lane's GPU (by UUID)             |
| `GET /api/monitor`                      | Latest thermals/power/VRAM + Ollama split                     |
| `GET /api/monitor/history?window=`      | Bucketed telemetry history over the last `window` seconds     |
| `GET /api/doctor`                       | On-box recon report (per-lane checks)                         |
| `GET /api/jobs/{id}`                    | Progress + log for a long load/pull/download                  |
| `GET /api/load-times`                   | Measured launch durations, every key (`{key: {est_s, n}}`)    |
| `GET /api/load-times/{model}`           | One model's expected load time per unit (+ residency/failures)|
| `GET /api/placement/decisions`          | Last ~50 auto-placement decisions: unit, tier, candidate facts|
| `POST /api/load`                        | `{server,model,lane?,force?,max_pack?}` → a Job               |
| `POST /api/unload`                      | `{server?,lane?}` → free that lane's GPU                      |
| `GET / PUT /api/lanes/{id}/default`     | Get / set a lane's startup-default model                      |
| `POST /api/ollama/pull`                 | Pull an Ollama model (job)                                    |
| `DELETE /api/ollama/{name}`             | Delete an Ollama model                                        |
| `GET/POST/PUT/DELETE /api/vllm/aliases` | The vLLM alias registry for a lane (add/edit/remove)          |
| `POST /api/vllm/download`               | HuggingFace-download a model into the WSL cache               |
| `POST /api/leases`                      | Claim a unit → a lease id (409 when it's held)                |
| `GET /api/leases`, `GET /api/leases/{id}` | List / poll leases (incl. why one ended)                    |
| `POST /api/leases/{id}/renew`           | Push the TTL out; `DELETE /api/leases/{id}` hands it back     |
| `POST /api/leases/{id}/revoke`          | Operator break-glass — take a unit back                       |

## Leases — sharing a unit between callers

`swap_in_progress` only answers *"is a swap running right now?"*. A **lease** is a
real claim: a caller says who it is, **whether its work may be interrupted**, and
**how long it needs the unit** — and finds out if it gets displaced.

```bash
LEASE=$(llmconfig lease claim --holder nightly-eval --no-preempt \
                              --minutes 10 --expect-minutes 45 --quiet)
curl -H "X-LLM-Lease: $LEASE" ... /v1/chat/completions     # your traffic
llmconfig lease renew  $LEASE                              # before the TTL lapses
llmconfig lease release $LEASE                             # hand it back
```

- **Two durations on purpose.** `--expect-minutes` is the honest "I'll need this
  for 45 minutes" (recorded, displayed, never enforced); `--minutes` is a short
  renewable leash proving you're alive. A crashed client frees the unit in minutes
  instead of pinning it for its whole declared duration.
- **Preemption.** A `--no-preempt` lease is never taken by another claim (not even
  with `--force`); a preemptible one yields to a higher `--priority` claim. Equal
  priority is first-come-first-served unless `--force`.
- **Being displaced** shows up two ways: `GET /api/leases/{id}` reports
  `state=revoked` with who took it and why, and your next `/v1` request returns
  `409 lease_revoked`. `llmconfig lease wait <id>` blocks and exits **0** released
  / **3** revoked / **4** expired, so a shell driver can react.
- **Effect on others.** Un-leased `/v1` traffic keeps working normally (opencode
  needs no changes) *except* against a non-preemptible lease, which refuses it with
  `409 lease_required`. The same rule guards `POST /api/load` and `POST /api/unload`
  (the holder passes by sending its lease id as `X-LLM-Lease`) — otherwise anyone
  could swap the model right over the held unit. Set `LEASE_BLOCK_UNLEASED=false`
  to disable both.
- **Effect on idle-unload.** Any live lease stops the [idle reaper](#idle-auto-unload),
  so a holder pausing between bursts keeps its model resident — at the cost of the
  card staying in P0 for up to one lease period.
- **Expiry releases the claim; it does not unload anything.** The model stays
  resident and the reaper simply resumes its normal timer.

> **Advisory, not enforced.** Clients that bypass LLMConfig (direct Ollama on
> `:11434`/`:11435`, the vLLM relay on `:11437`) are ungated, so a non-preemptible
> lease is a cooperation contract rather than a hard exclusivity guarantee. It is
> also a **forward** guarantee only: work already running when you claim keeps the
> unit until it finishes (there is no job cancellation) — the claim response reports
> that as `busy_with`.

## OpenAI `/v1` gateway — auto-load on first request

An OpenAI-compatible gateway on the same port (`http://192.168.1.40:11430/v1`) so a
client can switch models **without a manual swap**: the first request for a model
loads it. Built for opencode's `/model` picker (which has no selection-time hook), so
the switch happens on the inference path.

`GET /v1/models` · `POST /v1/chat/completions` · `POST /v1/completions` ·
`POST /v1/embeddings` · `POST /v1/rerank` · `POST /v1/score`.

- **Unit selection:** `X-LLM-Lane: <unit>` pins. **No header (or `auto`) auto-places**:
  the unit already serving the model wins (idle preferred, deterministic tie-break for
  prefix-cache affinity); else a unit with free capacity (declared Spark budgets, or a
  free lane); else one whose idle + unleased occupant can be evicted; else `503
  no_capacity` naming why each unit refused. The chosen unit is echoed back as
  **`X-LLM-Unit`**. A valid `X-LLM-Lease` pins to its lease's unit. Placement is
  advisory — admission, the unit locks and the lease gate stay the real gates — and a
  conflict is retried once on the runner-up. `AUTO_PLACE_ENABLED=false` restores the
  old implicit-`primary` default. `/v1/models` without a header lists the whole
  fleet's catalog union.
- **Proven-load gate (2026-07-28):** a fresh load is only *chosen* on a unit where
  the model has **launched successfully before** — a recorded load-time sample, or
  current residency (Sparks prove fleet-wide, identical hardware; GPU lanes prove
  per-unit). An unproven model answers `503` with *"never loaded successfully here —
  load it once explicitly"*: seed it once via `/api/load` with an explicit lane (or
  the UI) and auto-placement takes it from there. Sole-candidate pins and resident
  models are exempt (a pin behaves as an explicit header). Alongside it, a
  **failure blocklist**: `PLACEMENT_FAIL_BLOCK_AFTER` (default 2) consecutive
  launch failures skip that unit for `PLACEMENT_FAIL_BLOCK_COOLDOWN_S` (30 min),
  then allow one probe. `PLACEMENT_REQUIRE_PROVEN=false` disables the gate. Note
  the limit: *proven* means the launch succeeds, not that inference is stable.
- **Load-time estimates:** every successful launch records its duration;
  `GET /api/load-times` lists all keys, `GET /api/load-times/{model}` answers for
  one model per unit (`est_s: null` until the first load), and
  `llmconfig load-times [MODEL]` is the CLI view. Placement uses the estimates as
  a tier-3 tie-break — a fresh load goes where it comes up fastest (bucketed to
  the minute; ties fall back to emptiest-first).
- **Decision log:** `GET /api/placement/decisions` shows the last ~50 placements,
  newest first — which unit won, which tier fired (`pin`/`resident`/`fits`/
  `displace`), and per-candidate facts (proven, fail-blocked, lease-refused,
  committed budgets) for the losers. Consecutive identical routine placements
  collapse into one entry with a `count`; in-memory only, empty after a restart.
  This is the "why did that land on spark3?" surface — the always-on task's
  console is effectively write-only, so it's an endpoint, not a log line.
- **Resolution (per unit):** a Spark catalog `served_name` → that node's own slot
  port (multi-model nodes route per model); a vLLM `served_name` → the lane's relay;
  else an Ollama tag (has a `:`) → the lane's Ollama; else `404`.
- **No new arbitration** — it reuses `/api/load` (the per-lane lock, eviction-wait
  gate, WSL keepalive, Spark admission). On a **streaming** request it relays the
  cold-load progress as chat chunks (`⏳ …`), then forwards the real completion
  verbatim.
- **Edge cases:** identical concurrent loads coalesce onto one job; a non-stream chat
  request mid-load of a *different* model returns an empty `200` on a GPU lane (so
  title-gen never blocks) — a Spark queues instead, since the other load is another
  slot's business; pooling routes never fabricate (`503`, because an empty embedding
  written to a vector store is silent corruption); cold-load timeout → `503`.

Point a provider's `baseURL` at `http://192.168.1.40:11430/v1`. The always-on app must
be **restarted** to pick up a new gateway build.

## Web UI

`http://192.168.1.40:11430/` — tabs are built at runtime from `/api/lanes`, so a unit
appearing in config appears in the UI with no front-end change.

| Tab             | What it is                                                                             |
| --------------- | -------------------------------------------------------------------------------------- |
| **Home**        | One box per unit: owner badge, loaded model + its real HF repo, served context, memory bar, and a **quick-switch dropdown**. Clicking a unit's name jumps to its tab. |
| **Per unit**    | One tab each for the 3090, 3070 Ti and every Spark. GPU lanes show the Ollama ⇄ vLLM split with per-model Load / star-as-default buttons and an Ollama pull box; Spark tabs show the curated recipe list. |
| **Monitor**     | Every unit's thermals, power, memory and the Ollama GPU/CPU split, over a selectable window. |

Units that are configured but unreachable render greyed rather than disappearing, so a
powered-off Spark is visibly *offline* instead of silently absent. Load/unload state is
tracked **per unit**, so a 15-minute Spark load never freezes the other five boxes.

A shared **Activity** drawer at the bottom streams the log of whichever job is running.

## CLI

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
activity, letting the card fall to P8. Participation is per lane: the **companion
3070 Ti is exempt by default** (`COMPANION_IDLE_UNLOAD_ENABLED=false`) — it reaches P8
(~13 W) even with a small model resident, so reaping it saves ~nothing and would cost
its always-resident relay model the instant response; flip the flag to opt it in.

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

The same activity signals back a **usage query**: `GET /api/usage` (and the `usage`
field on each `/api/status` lane, plus `llmconfig usage`) classifies every lane as
`free` (nothing loaded), `idle` (model loaded but unused — the name is returned), or
`active` (model loaded and in use). "Active" means activity within
`USAGE_ACTIVE_WINDOW_S` (default 60 s), a currently-visible GPU-utilization sample, or
a swap in flight.

## Context windows

Every unit reports the context it is **actually serving at** — surfaced on each Home
box and on `/api/status` → `lanes[].loaded.context_len`. This is deliberately the
*runtime* window, not the model's architectural maximum, because it is the number a
client's prompt budget has to respect:

| Backend | Source                        | Set by                                     |
| ------- | ----------------------------- | ------------------------------------------ |
| Ollama  | `/api/ps` → `context_length`  | the tag's baked `num_ctx`, capped by `OLLAMA_CONTEXT_LENGTH` |
| vLLM    | relay `/v1/models`            | `--max-model-len` in that alias's `serve.sh` case |
| Spark   | the node's `/v1/models`       | `--max-model-len`, pinned from the catalog entry  |

**Ollama takes no context parameter at load time.** `/api/load` and the `/v1` auto-load
deliberately do not accept one — a model loads at the `num_ctx` baked into its
Modelfile, so the KV-cache footprint is fixed by the *tag*, not the caller. To change
it, bake a tag:

```bash
# Modelfile:  FROM <model>  +  PARAMETER num_ctx 65536
ollama create <model>-64k -f Modelfile    # reuses the existing weights blob
llmconfig load ollama <model>-64k
```

> `OLLAMA_CONTEXT_LENGTH` (per NSSM service, in `AppEnvironmentExtra`) **silently
> truncates** any model without a baked `num_ctx`. An *absent* variable is the easy
> miss — it falls back to Ollama's built-in 4096 and looks identical to a correct
> setting until you read `/api/ps`. Both services on `.40` are at 32768.

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
- **A unit looks "busy" for hours and loads queue behind it** → it is *wedged*, not
  busy. Check `llmconfig doctor` for `wsl.distro` ✗ and `wsl.selfheal`. Since
  2026-07-28 loads and unloads acquire the unit lock with a timeout and fail naming
  the blocking job, so this surfaces instead of stacking (29 jobs stacked, once).
- **Everything WSL-side hangs after a reboot** → the distro's *exec* path is wedged.
  Diagnose by contrast: `wsl --status` answers normally while `wsl -u <user> -- echo hi`
  hangs. `WslRecovery` runs automatically (kill orphans → `--shutdown` → restart
  `WslService`); by hand, note that **`wsl --shutdown` alone is not enough** — it can
  time out without reaping `vmmemWSL`, and only `Restart-Service WslService -Force`
  clears it.

## Surviving reboots

The box patches itself and Windows Update may auto-restart it unattended (permitted
window 00:00-06:00). LLMConfig is expected to come back **clean, with no keyboard**:

| Layer | What it does |
| ----------------------------- | ----------------------------------------------- |
| Auto-logon (LSA secret)       | Guarantees the interactive session the task needs — the trigger is *at logon*, and the app must run as a real user because `wsl.exe -u <user>` has no Session 0 equivalent |
| Task delay `PT2M` + `RestartCount 3` | Keeps the app off a cold WSL, and restarts it if it crashes |
| `wsl.wait_ready()` gate       | The real fix — the boot `autoload_defaults()` waits until WSL can actually *execute*, not merely respond |
| `WslRecovery`                 | Self-heals a distro that comes up wedged |
| Bounded locks + bounded reap  | A stuck operation fails and names itself instead of hanging the unit forever |

Without the gate, the app starting ~58 s after boot raced WSL's first-boot init and
deadlocked its exec path — a 60-second reboot became a six-hour outage.

## Project layout

```
llmconfig/
  config.py         Settings + LaneConfig/SparkConfig (all box-specific values)
  main.py           FastAPI app: REST endpoints + static UI + lifespan
  orchestrator.py   coordinates every unit; shared WSL keepalive + defaults
  lane.py           per-GPU arbitration state machine (eviction-wait gate)
  spark_unit.py     a remote DGX Spark as a unit (same contract, no eviction gate)
  leases.py         LeaseManager + LeaseSweeper (resource sharing between callers)
  lane_state.py     persisted per-unit default model
  backends/
    ollama.py       Ollama REST client + Windows service control
    vllm.py         vLLM relay status + serve.sh/systemctl lifecycle over wsl.exe
    spark.py        sparkrun over WSL + node HTTP status + remote nvidia-smi
  gpu.py            nvidia-smi truth (by UUID) + Monitor metric sampling
  nvapi.py          NVAPI hotspot + GDDR6X junction temps (ctypes)
  monitor.py        telemetry sampler + SQLite history
  idle.py           idle auto-unload policy (reap an unused unit → GPU drops to P8)
  registry.py       vLLM alias + Spark recipe catalogs (YAML)
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

**Live-verified on `.40`.**

- **GPU lanes** — `doctor --local` green; both paths exercised end to end on the 3090
  (Ollama load/unload; vLLM load that evicts → packs VRAM → serves through the relay →
  unloads to 0 MiB). The 3070 Ti lane is proven for Ollama; companion vLLM is optional
  and installed separately.
- **DGX Sparks** — first cluster load **2026-07-24**: `gemma-4-26b-fp8` on all four
  nodes, verified with real completions through the `/v1` gateway (0.6–1.1 s), ~85–90 %
  of each 122 GB unified pool resident. Weights are staged between nodes over the LAN
  with a doubling tree rather than downloaded per node.
- **Leases** — full lifecycle proven against live inference: un-leased traffic allowed →
  non-preemptible claim → `409 lease_required` → holder passes with `X-LLM-Lease` →
  revoke → `409 lease_revoked`.
- **Telemetry** persisted across restarts.

See the homelab wiki (`hosts/ollama-host/services/llmconfig`) for deployed-state detail
and `runbooks/local-llm-server-dgx-spark` for the cluster.

## License

MIT.
