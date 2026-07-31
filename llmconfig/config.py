"""Runtime configuration, loaded from environment / `.env` via pydantic-settings.

Every box-specific value lives here so the app can be retargeted at a different
host (or the live `.40` specifics confirmed via `llmconfig doctor`) without code
changes. Defaults match the documented `Alien-3070-TI` setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the `llmconfig/` package dir. Used for default data paths.
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LaneConfig:
    """Everything that pins one inference lane to one GPU. The `Orchestrator` runs
    one `Lane` per `LaneConfig`; the primary lane is the RTX 3090, the optional
    companion lane is the RTX 3070 Ti."""

    id: str                       # "primary" | "companion"
    name: str                     # display label, e.g. "RTX 3090"
    gpu_uuid: str
    vram_total_mb: int
    vram_free_baseline_mb: int
    ollama_url: str
    ollama_service_name: str
    vllm_relay_url: str
    vllm_serve_script: str
    vllm_systemd_unit: str
    registry_path: Path
    enabled: bool = True
    # Does this lane have a working vLLM half? A lane is Ollama-XOR-vLLM by
    # design, but the vLLM side needs a serve script + systemd unit installed on
    # the box, and the COMPANION's was specified and never built (no
    # `serve-companion.sh` exists — not on .40, not in deploy/). False makes that
    # explicit instead of pretending: doctor reports it as configuration rather
    # than failure, the catalogs stop advertising models that cannot run, and a
    # load refuses BEFORE it evicts the lane's working Ollama model.
    vllm_enabled: bool = True
    default_server: str = ""      # "ollama" | "vllm" | "" — auto-load on startup
    default_model: str = ""       # Ollama tag or vLLM alias
    # Whether the idle reaper may unload this lane (the global idle_unload_enabled
    # is the master switch; this is per-lane participation).
    idle_unload_enabled: bool = True
    kind: str = "gpu"


@dataclass(frozen=True)
class SparkConfig:
    """One remote NVIDIA DGX Spark (GB10) node, driven by `sparkrun` over WSL.

    Unlike a `LaneConfig` there is no intra-unit arbitration and no eviction-wait
    gate: memory is released when a container stops, so there is nothing to poll
    for. The node can host SEVERAL models at once (128 GB unified), each as its
    own sparkrun workload on its own port `api_port + slot`; co-residency is
    bounded by each recipe's declared `mem_fraction`, not by a hardware gate.
    Status is read over HTTP from each slot's OpenAI endpoint; lifecycle goes
    through sparkrun.
    """

    id: str                       # "spark1".."spark4" — the API/lane key
    name: str                     # display label, e.g. "spark-cc9b"
    host: str                     # LAN address, e.g. "192.168.1.50"
    ssh_user: str                 # SSH user on the node (for remote nvidia-smi)
    api_port: int                 # base port; slot N serves on api_port + N
    registry_path: Path           # curated per-node model catalog
    max_models: int = 4           # concurrent workloads => ports api_port..+max_models-1
    enabled: bool = True
    # GB10 is 128 GB unified memory; nvidia-smi reports memory.total as [N/A] on
    # these parts, so this is the fallback denominator for the VRAM percentage.
    vram_total_mb: int = 122880   # ~120 GiB usable
    default_model: str = ""       # curated alias to auto-load on startup
    # Remote nodes idle at ~25 W and reloading costs minutes (weights are large),
    # so Sparks are exempt from the idle reaper by default.
    idle_unload_enabled: bool = False
    load_timeout_s: int = 900
    kind: str = "spark"

    @property
    def api_base(self) -> str:
        """Slot 0. Kept as a property because plenty of call sites want *an*
        endpoint for the node; anything routing a specific model must use
        `api_base_for(port)` instead."""
        return f"http://{self.host}:{self.api_port}"

    def api_base_for(self, port: int) -> str:
        """The endpoint a model served on `port` is reachable at."""
        return f"http://{self.host}:{port}"

    @property
    def slot_ports(self) -> tuple[int, ...]:
        """Every port a workload on this node may occupy. `status()` probes all of
        them, which is how the port->model map is discovered rather than stored —
        so it survives an LLMConfig restart with models already resident."""
        return tuple(self.api_port + i for i in range(max(1, self.max_models)))

    @property
    def gpu_uuid(self) -> str:
        """Synthetic telemetry key. Remote GB10s aren't in the local nvidia-smi
        namespace, so Monitor/idle lookups key off this instead of a real UUID."""
        return f"spark:{self.id}"


@dataclass(frozen=True)
class SparkGroupConfig:
    """A SET of Spark nodes serving ONE tensor-parallel model over the 200G fabric.

    Synthetic and UI-invisible: a `SparkGroup` built from this lives in
    `orch.units` (so placement, leases, and the /v1 gateway treat it like any
    unit) but is never emitted by `settings.units()` — the UI's tab/card source —
    and is filtered out of `/api/status` lanes. The members' own cards carry the
    residency (`LoadedModel.group`).

    The id is the sorted member ids joined with "_" (`spark1_spark2`): URL-safe
    ("+" would decode to a space in query strings), collision-free with node ids
    and the reserved "auto". The HEAD is the first member in settings order — a
    tp job serves /v1 from one rank, and that is the endpoint `status()` probes
    and the gateway routes to.
    """

    id: str                        # "spark1_spark2"
    name: str                      # display label, e.g. "Sparks 1+2"
    member_ids: tuple[str, ...]    # sorted member unit ids
    head_host: str                 # the head member's LAN address
    api_port: int                  # the head's base port — the group's serve port
    enabled: bool = True
    max_models: int = 1            # a tp job claims every member whole
    # A group has no Ollama/vLLM halves. Declared explicitly (rather than left
    # absent) so any code path that reaches for the lane shape degrades to
    # "not available" instead of raising AttributeError.
    vllm_enabled: bool = False
    default_model: str = ""        # groups are never auto-loaded at startup
    # Groups are never idle-reaped in v1 (same rationale as single Sparks, ×K:
    # reloading costs minutes per node and idle nodes draw ~25 W).
    idle_unload_enabled: bool = False
    # Multi-node cold starts move hundreds of GB and synchronize K ranks; give
    # them far more rope than a single node's default.
    load_timeout_s: int = 3600
    kind: str = "spark_group"

    def api_base_for(self, port: int) -> str:
        """The endpoint the group's model is reachable at — always on the HEAD.
        Signature-compatible with SparkConfig so the gateway's `_route` needs no
        special case."""
        return f"http://{self.head_host}:{port}"

    @property
    def api_base(self) -> str:
        return self.api_base_for(self.api_port)

    @property
    def gpu_uuid(self) -> str:
        """Synthetic telemetry key, mirroring SparkConfig's convention."""
        return f"sparkgroup:{self.id}"


def group_id_for(member_ids: list[str] | tuple[str, ...]) -> str:
    """The canonical group id for a node set: sorted, joined with '_'."""
    return "_".join(sorted(member_ids))


def _parse_fabric_links(spec: str) -> list[frozenset[str]]:
    """Parse `SPARK_FABRIC_LINKS` → the node sets that are physically cabled.

    Format: comma-separated groups, members joined with `+`, e.g.
        spark1+spark2,spark3+spark4
    for two directly-cabled pairs. One group naming every node
    (`spark1+spark2+spark3+spark4`) is what a switched fabric looks like.

    Empty/unset returns `[]`, which means UNCONSTRAINED — every node set is
    allowed, exactly as before this setting existed. That is deliberate: the
    constraint is opt-in, so an operator who has not described their cabling
    keeps the old behaviour rather than silently losing the Cluster tab.

    Malformed entries are skipped rather than raising, matching
    `_parse_spark_nodes`: a typo in .env must not stop the app from starting.
    Single-member groups are dropped too — a "link" of one cables nothing and
    would only ever reject every multi-node set.
    """
    links: list[frozenset[str]] = []
    for chunk in (spec or "").split(","):
        members = {p.strip() for p in chunk.split("+") if p.strip()}
        if len(members) >= 2:
            links.append(frozenset(members))
    return links


def _parse_spark_nodes(spec: str) -> list[tuple[str, str, str]]:
    """Parse `SPARK_NODES` → [(id, host, name)].

    Format: comma-separated `id=host[=name]`, e.g.
        spark1=192.168.1.50=spark-cc9b,spark2=192.168.1.51=spark-4cd0
    Malformed entries are skipped rather than raising — a typo in .env must not
    stop the app from starting with the local lanes.
    """
    nodes: list[tuple[str, str, str]] = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("=")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        if parts[0].lower() == "auto":
            continue  # reserved: the gateway's auto-placement sentinel, never a unit id
        node_id, host = parts[0], parts[1]
        name = parts[2] if len(parts) > 2 and parts[2] else node_id
        nodes.append((node_id, host, name))
    return nodes


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- the control app itself ---
    llmconfig_host: str = "0.0.0.0"
    llmconfig_port: int = 11430
    llmconfig_api_key: str = ""  # optional; protects write ops when non-empty

    # --- Ollama (Windows-native) ---
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_service_name: str = "ollama"

    # --- vLLM (WSL2, reached via the socat relay) ---
    vllm_relay_url: str = "http://127.0.0.1:11437"  # 127.0.0.1, not localhost
    vllm_serve_script: str = "/home/folar/vllm/serve.sh"
    vllm_systemd_unit: str = "vllm@"  # templated user unit; instance = alias
    # `hf` lives in this venv, NOT on the WSL login PATH — the download endpoint
    # sources this before `hf download` (a bare `hf` fails "command not found").
    vllm_venv_activate: str = "/home/folar/vllm/.venv/bin/activate"

    # --- WSL plumbing ---
    wsl_distro: str = "Ubuntu-24.04"
    wsl_user: str = "folar"

    # --- GPU (primary lane = RTX 3090) ---
    gpu_uuid: str = "GPU-739bece9-8298-7993-f7dd-c8d86cb541f9"  # the RTX 3090
    vram_total_mb: int = 24576
    vram_free_baseline_mb: int = 1500  # "freed" / "maxed" threshold

    # --- companion lane (RTX 3070 Ti) — optional second GPU, off by default ---
    companion_enabled: bool = False
    companion_gpu_uuid: str = "GPU-2caf7863-102e-31e5-be4d-5ec860addc78"  # the RTX 3070 Ti
    companion_vram_total_mb: int = 8192
    companion_vram_free_baseline_mb: int = 600
    companion_ollama_url: str = "http://127.0.0.1:11435"        # 2nd Ollama instance
    companion_ollama_service_name: str = "OllamaCompanion"
    companion_vllm_relay_url: str = "http://127.0.0.1:11438"    # 2nd socat relay
    # OFF: `serve-companion.sh` has never existed — the unit file specifies it
    # (resolve the 3070 Ti index by UUID via torch, lower --gpu-memory-utilization)
    # but the script was never written, so the companion is an Ollama-only lane.
    # Flip to true if you ever build it; the registry + unit are already here.
    companion_vllm_enabled: bool = False
    companion_vllm_serve_script: str = "/home/folar/vllm/serve-companion.sh"
    companion_vllm_systemd_unit: str = "vllm-companion@"
    companion_registry_path: Path = REPO_ROOT / "data" / "vllm_models_companion.yaml"
    companion_default_server: str = ""   # "ollama" | "vllm" | "" — auto-load on startup
    companion_default_model: str = ""    # Ollama tag or vLLM alias

    # --- DGX Spark cluster (4× GB10, driven by sparkrun from this box's WSL) ---
    # Off by default: enable once sparkrun is installed and the `sparks` cluster is
    # configured on the box (see runbooks/local-llm-server-dgx-spark on the wiki).
    spark_enabled: bool = False
    # `id=host[=name]`, comma separated. Defaults to the as-built site-A cluster.
    spark_nodes: str = (
        "spark1=192.168.1.50=spark-cc9b,"
        "spark2=192.168.1.51=spark-4cd0,"
        "spark3=192.168.1.52=spark-b984,"
        "spark4=192.168.1.53=spark-f04a"
    )
    spark_ssh_user: str = "fksogbetun"
    spark_api_port: int = 8000          # base OpenAI port; slot N uses base + N
    # Concurrent models per node. A GB10 holds 128 GB unified, so several models fit
    # if each declares a `mem_fraction`. Each slot costs one HTTP probe per status
    # poll (they run concurrently), so this is the knob that trades poll cost for
    # capacity.
    # --- auto-placement (the /v1 gateway picks the unit when none is named) ---
    # Kill switch: off restores the pre-placement behavior byte-for-byte (a request
    # without X-LLM-Lane lands on "primary").
    auto_place_enabled: bool = True
    # TTL on the placer's unit-status sweep; staleness only mis-ranks (admission,
    # the lane lock, and the lease gate are the real gates).
    placement_cache_ttl_s: float = 2.0
    # Proven-load gate: auto-placement only CHOOSES a unit where this model has
    # launched successfully before (a LoadTimes sample, or current residency —
    # Sparks prove fleet-wide, lanes per-unit). The tiers that don't choose are
    # exempt: a sole-candidate pin behaves as an explicit header (that's how a
    # first-ever load gets seeded), and a resident model is its own proof.
    placement_require_proven: bool = True
    # Consecutive launch failures on ONE unit before placement stops choosing it
    # (0 disables). After the cooldown one probe attempt is allowed; the counter
    # survives, so another failure re-blocks. A success clears it.
    placement_fail_block_after: int = 2
    placement_fail_block_cooldown_s: float = 1800.0
    # TTL on the per-lane Ollama tag list used by placement resolution — its only
    # real I/O. Tags change on pull timescales; staleness only mis-ranks. Also a
    # negative cache: a down Ollama answers "no tags" for the TTL instead of
    # stacking its client timeout onto every auto request for a tag.
    placement_tags_ttl_s: float = 15.0
    # Workload tiering: the 3090 is the SPEED tier (single-stream latency), the
    # Sparks are the CAPACITY tier (121 GB pools + continuous batching). When a
    # model resolves on both, the request's shape biases the choice: interactive
    # (small prompt AND bounded generation, thresholds below) prefers the GPU
    # lane; batch/bulk (long prompt, big max_tokens, or a pooling body) prefers
    # a Spark even over an idle lane. `X-LLM-Workload: interactive|batch`
    # overrides the heuristic. A preference only — never disqualifies a unit.
    placement_workload_enabled: bool = True
    placement_interactive_max_prompt_chars: int = 16000   # ~4k tokens at 4 chars/tok
    placement_interactive_max_new_tokens: int = 2048

    spark_max_models: int = 4
    # Total share of a node's pool that may be committed at once. Loads are refused
    # when the sum of resident `mem_fraction` plus the incoming model exceeds this.
    # Below 1.0 because vLLM's own allocation is not the only consumer on the node.
    spark_mem_headroom: float = 0.95
    spark_cluster: str = "sparks"       # saved sparkrun cluster name
    spark_vram_total_mb: int = 122880   # ~120 GiB usable of the 128 GB unified pool
    spark_load_timeout_s: int = 900     # weights are large; cold starts take minutes
    spark_idle_unload_enabled: bool = False
    # sparkrun command templates. Kept configurable because the exact flags differ
    # across sparkrun releases — correct them in .env rather than patching code.
    # Placeholders: {cluster} {host} {recipe} {tp} {port} {served} {user} {extra}
    #
    # Verified against sparkrun 0.2.40 on .40 (2026-07-24). Three details are
    # load-bearing, all confirmed by `--dry-run` on the live cluster:
    #   --cluster AND --hosts together: the cluster supplies the SSH user, --hosts
    #     narrows to one node. With --hosts alone sparkrun falls back to the local
    #     WSL user and every connection fails "folar@…: Permission denied".
    #   --no-follow: without it `run` tails the container logs and never returns,
    #     so the load would sit until its timeout instead of reporting success.
    #   --port/--served-model-name: pin what the node serves to what this app
    #     probes, so readiness and the /v1 resolver can't drift from the recipe.
    spark_run_cmd: str = (
        "sparkrun run {recipe} --cluster {cluster} --hosts {host} --tp {tp} "
        "--port {port} --served-model-name {served} --no-follow {extra}"
    )
    # `stop` requires a TARGET or --all; without either it exits "Must specify
    # TARGET or --all." and nothing is stopped.
    spark_stop_cmd: str = "sparkrun stop --all --cluster {cluster} --hosts {host}"
    # Stop ONE workload, leaving co-residents running. `stop` accepts a TARGET that
    # is a recipe name or a cluster id; the recipe is what this app already knows.
    # Without this every load would keep using `--all` and evict the neighbours it
    # is supposed to coexist with.
    # DO NOT stop by recipe name: `sparkrun stop <recipe> --cluster --hosts` prints
    # "Workload stopped" with rc=0 while stopping NOTHING (verified live 2026-07-26 —
    # the embedder kept serving through three "successful" stops). Only the job id
    # printed by `sparkrun status` actually stops a workload; stop also swallows
    # SSH failures into rc=0, so callers must verify via the slot probe, never
    # trust the exit code.
    spark_stop_one_cmd: str = "sparkrun stop {recipe} --cluster {cluster} --hosts {host}"  # broken; kept for reference
    spark_stop_job_cmd: str = "sparkrun stop {job} --cluster {cluster}"
    spark_status_cmd: str = "sparkrun status --cluster {cluster}"
    # --- multi-node (SparkGroup) launches over the 200G fabric ---
    # Master gate. False (the default) makes the whole multi-node feature inert:
    # no SparkGroup units exist, placement behaves byte-for-byte as before, and
    # POST /api/cluster/load refuses — the Cluster tab still renders as a planner.
    # Flip to true once the 200G switch is installed and `sparkrun setup cx7` has
    # brought the fabric up.
    spark_fabric_enabled: bool = False
    # WHICH nodes are actually cabled to each other. The fabric flag says a fabric
    # exists; this says what shape it is, and the two are independent — a fabric
    # can be up while only some nodes can reach each other.
    #
    # Comma-separated groups, members joined with '+':
    #     spark1+spark2,spark3+spark4      two directly-cabled pairs
    #     spark1+spark2+spark3+spark4      one switched fabric, any-to-any
    #
    # A multi-node launch is refused unless its node set fits INSIDE one group
    # (subset, not equality — on a 4-node switched fabric a 2-node job is fine).
    # Empty (the default) = unconstrained, i.e. byte-for-byte the behaviour from
    # before this setting, so nothing changes for a deployment that never sets it.
    #
    # Why this must exist: with two direct pairs, addresses are handed out from
    # one subnet across all four nodes, so spark1 and spark3 look same-subnet but
    # share no wire. A tp job spanning them does not error — it HANGS, because
    # Ray/NCCL peer discovery waits on a peer that cannot answer. Encoding the
    # cabling here is what turns that silent hang into a 400 with a readable
    # message. Measured 2026-07-30: within a pair, 196 Gb/s RDMA / 19.5 GB/s NCCL.
    spark_fabric_links: str = ""
    # The multi-host launch template. {hosts} = comma-joined member addresses in
    # member order (head first); {tp} = the member count K. Mirrors spark_run_cmd
    # otherwise — same load-bearing flags, same reasons.
    # ⚠ UNVERIFIED against a live multi-node sparkrun (the fabric isn't up to
    # --dry-run against): comma-joined --hosts is the documented form, but re-verify
    # with `sparkrun run <recipe> --dry-run --cluster sparks --hosts h1,h2 --tp 2`
    # before the first real launch, and correct THIS SETTING in .env if it moved.
    spark_run_multi_cmd: str = (
        "sparkrun run {recipe} --cluster {cluster} --hosts {hosts} --tp {tp} "
        "--port {port} --served-model-name {served} --no-follow {extra}"
    )
    # The cluster-wide catalog of multi-node recipes (min_nodes > 1). Deliberately
    # SEPARATE from the per-node catalogs: a multi-node model must resolve only on
    # SparkGroup units and a single-node model only on individual nodes — keeping
    # the two lists disjoint is what makes that partition hold with no ranking
    # special-cases.
    spark_cluster_registry_path: Path = REPO_ROOT / "data" / "spark_cluster_models.yaml"
    # Where successful (model, node-set) placements are recorded — the memory
    # behind "loaded once ⇒ listed on those cards and available to auto-placement".
    spark_group_state_path: Path = REPO_ROOT / "data" / "spark_group_state.yaml"
    # Remote telemetry: plain SSH to the node (the control node's WSL already has
    # passwordless key auth to every Spark).
    spark_ssh_cmd: str = "ssh -o BatchMode=yes -o ConnectTimeout=5 {user}@{host} {command}"

    # --- monitoring (the Monitor tab: thermals/power/VRAM history) ---
    monitor_enabled: bool = True
    monitor_interval_s: float = 5.0   # GPU sample cadence
    monitor_retention_h: int = 24     # history window (in-memory + on-disk)
    # Persist samples to SQLite so the history survives an app/service restart.
    # When false, history is in-memory only (lost on restart, as before).
    monitor_persist: bool = True
    monitor_db_path: Path = REPO_ROOT / "data" / "monitor.db"

    # --- idle auto-unload (power: reap an idle lane so the card drops to P8) ---
    # A resident model pins the card in P0 (~117 W on the 3090); unloading lets it
    # fall to its ~25 W P8 idle. Activity = a /v1 gateway request, a load finishing,
    # or a Monitor utilization sample above the threshold (the last catches clients
    # that talk to Ollama / the vLLM relay directly, bypassing the gateway).
    idle_unload_enabled: bool = True
    idle_unload_after_min: float = 15.0       # sustained inactivity before reaping
    idle_unload_check_interval_s: float = 60.0
    idle_unload_util_pct: float = 5.0         # util above this counts as activity
    # The companion 3070 Ti is EXEMPT by default: it reaches P8 (~13 W) even with a
    # small model resident, so reaping saves ~nothing and would cost the opencode
    # /swap echo relay its instant response. Opt it in explicitly if that changes.
    companion_idle_unload_enabled: bool = False
    # The primary 3090 has its own knob so a pinned daily-driver model (vl32) can be
    # exempted without turning the reaper OFF globally — the global switch would also
    # stop the per-tick Monitor folding that keeps `idle_s`/usage honest fleet-wide.
    primary_idle_unload_enabled: bool = True
    # Recent-activity window for classifying a loaded lane "active" (GET /api/usage
    # and the `usage` field on /api/status lanes).
    usage_active_window_s: float = 60.0

    # --- leases (resource sharing between callers — see leases.py) ---
    # A lease is a caller's claim on one unit: "I'm using this, and here's whether
    # you may take it from me." Advisory by design — clients that bypass LLMConfig
    # (direct Ollama on :11434/:11435) can't be stopped, so a non-preemptible lease
    # is a cooperation contract, not a hard exclusivity guarantee.
    lease_enabled: bool = True
    # `X-LLM-Hold: <holder>` — a client that cannot carry a dynamic lease id (a
    # static config like opencode's) asks the gateway to hold its model for it.
    # PREEMPTIBLE on purpose: it shields the model from AUTOMATIC displacement (the
    # idle reaper, placement eviction) without 409-ing anyone else's traffic. Each
    # request renews it, so it lapses this long after the client stops talking.
    auto_hold_enabled: bool = True
    auto_hold_ttl_s: float = 600.0
    lease_default_ttl_s: float = 600.0     # renewable leash when the claim omits ttl_s
    lease_min_ttl_s: float = 30.0
    lease_max_ttl_s: float = 7200.0        # claims are CLAMPED to this, never refused
    # Past expiry a lease immediately stops blocking others, but for this long its
    # own requests still succeed and `renew` can revive it — absorbs a holder whose
    # renew timer slipped behind a long generation.
    lease_expiry_grace_s: float = 30.0
    lease_sweep_interval_s: float = 5.0
    lease_max_history: int = 100           # terminal leases kept for the "why" poll
    # Any live lease blocks the idle reaper, so a holder pausing between bursts keeps
    # its model resident. Cost: the card stays in P0 for up to one lease period.
    lease_blocks_idle_unload: bool = True
    # Whether a non-preemptible lease 409s un-leased /v1 traffic. The kill switch if
    # it ever surprises a client: false ⇒ leases only affect the reaper and other claimants.
    lease_block_unleased: bool = True
    # Auto-expire a lease whose holder never used it and whose unit sits free (a
    # claim whose load failed would otherwise 409 traffic on an empty card). 0 disables.
    lease_unused_release_s: float = 300.0
    # How to refuse a rejected `stream: true` request. "http" (a real 409) is correct
    # because the gate decides before any bytes are written; "sse" emits an in-band
    # error chunk at HTTP 200 for clients that can't surface a non-200 on a stream.
    lease_stream_reject_mode: str = "http"

    # --- HuggingFace (vLLM downloads) ---
    hf_token: str = ""

    # --- paths ---
    registry_path: Path = REPO_ROOT / "data" / "vllm_models.yaml"

    # --- timeouts / tuning (seconds) ---
    http_timeout_s: float = 10.0
    # Liveness probe to the (WSL) vLLM relay. When the relay is down, WSL2
    # localhost-forwarding blackholes the SYN (no RST), so the probe hangs ~2.4s;
    # cap it so /api/status stays snappy. The relay answers in ms when it's up.
    vllm_probe_timeout_s: float = 1.0
    evict_timeout_s: float = 45.0
    poll_interval_s: float = 2.0
    default_vllm_load_timeout_s: int = 240
    vllm_ready_grace_s: int = 30  # readiness re-check after a load's per-alias timeout, so a
                                  # vLLM that came up just past the deadline isn't failed/torn down

    # --- WSL readiness / recovery ---
    # At boot the app starts seconds after logon, while WSL2 is still coming up.
    # Firing autoload_defaults() into a cold distro deadlocks the exec path:
    # `wsl --status` answers but `wsl -u <user>` never returns, and every later
    # load inherits the wedge (2026-07-28: 6 h outage, 29 stacked jobs).
    wsl_ready_timeout_s: float = 300.0        # give up gating the boot autoload after this
    wsl_ready_probe_timeout_s: float = 15.0   # per-probe budget for `wsl -- true`
    wsl_ready_backoff_s: float = 5.0          # pause between probes
    # Self-heal: after this many consecutive probe timeouts, run the recovery
    # ladder (kill orphans -> --shutdown -> restart WslService).
    wsl_selfheal_enabled: bool = True
    wsl_selfheal_after_failures: int = 3
    wsl_selfheal_cooldown_s: float = 900.0    # never loop the ladder faster than this
    wsl_service_restart_timeout_s: float = 120.0  # observed ~16 stop-poll cycles

    # Bounded wait for a unit's swap lock. Must exceed the longest legitimate
    # queue (spark_load_timeout_s 900 / default_vllm_load_timeout_s 240), but
    # bounded so a wedged holder surfaces instead of queueing forever.
    swap_wait_timeout_s: float = 1200.0

    @property
    def auth_enabled(self) -> bool:
        return bool(self.llmconfig_api_key.strip())

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.llmconfig_host in ("0.0.0.0", "") else self.llmconfig_host
        return f"http://{host}:{self.llmconfig_port}"

    def lanes(self) -> list[LaneConfig]:
        """The lanes to run: always the primary (RTX 3090); the companion (RTX 3070
        Ti) when `companion_enabled`."""
        lanes = [
            LaneConfig(
                id="primary",
                name="RTX 3090",
                gpu_uuid=self.gpu_uuid,
                vram_total_mb=self.vram_total_mb,
                vram_free_baseline_mb=self.vram_free_baseline_mb,
                ollama_url=self.ollama_url,
                ollama_service_name=self.ollama_service_name,
                vllm_relay_url=self.vllm_relay_url,
                vllm_serve_script=self.vllm_serve_script,
                vllm_systemd_unit=self.vllm_systemd_unit,
                registry_path=self.registry_path,
                enabled=True,
                idle_unload_enabled=self.primary_idle_unload_enabled,
            ),
        ]
        if self.companion_enabled:
            lanes.append(
                LaneConfig(
                    id="companion",
                    name="RTX 3070 Ti",
                    gpu_uuid=self.companion_gpu_uuid,
                    vram_total_mb=self.companion_vram_total_mb,
                    vram_free_baseline_mb=self.companion_vram_free_baseline_mb,
                    ollama_url=self.companion_ollama_url,
                    ollama_service_name=self.companion_ollama_service_name,
                    vllm_relay_url=self.companion_vllm_relay_url,
                    vllm_serve_script=self.companion_vllm_serve_script,
                    vllm_systemd_unit=self.companion_vllm_systemd_unit,
                    registry_path=self.companion_registry_path,
                    enabled=True,
                    vllm_enabled=self.companion_vllm_enabled,
                    default_server=self.companion_default_server,
                    default_model=self.companion_default_model,
                    idle_unload_enabled=self.companion_idle_unload_enabled,
                )
            )
        return lanes

    def sparks(self) -> list[SparkConfig]:
        """The DGX Spark nodes to expose as units — empty unless `spark_enabled`."""
        if not self.spark_enabled:
            return []
        out: list[SparkConfig] = []
        for idx, (node_id, host, name) in enumerate(_parse_spark_nodes(self.spark_nodes), start=1):
            # Display as "Spark 1 (spark-cc9b)": the ordinal is what a human uses
            # day to day, the chassis hostname is what identifies the physical box
            # in NetBox and in `sparkrun status`, so keep both.
            digits = "".join(ch for ch in node_id if ch.isdigit())
            ordinal = int(digits) if digits else idx
            label = f"Spark {ordinal}" + (f" ({name})" if name and name != node_id else "")
            out.append(
                SparkConfig(
                    id=node_id,
                    name=label,
                    host=host,
                    ssh_user=self.spark_ssh_user,
                    api_port=self.spark_api_port,
                    max_models=self.spark_max_models,
                    registry_path=REPO_ROOT / "data" / f"spark_models_{node_id}.yaml",
                    vram_total_mb=self.spark_vram_total_mb,
                    idle_unload_enabled=self.spark_idle_unload_enabled,
                    load_timeout_s=self.spark_load_timeout_s,
                )
            )
        return out

    def units(self) -> list[LaneConfig | SparkConfig]:
        """Every LLM unit in display order: local GPU lanes first, then Sparks.

        This is the list the UI turns into tabs and dashboard cards; `lanes()` and
        `sparks()` remain available for code that needs only one kind. SparkGroups
        are deliberately ABSENT — they are synthetic orchestrator-level units with
        no tab or card of their own (see SparkGroupConfig).
        """
        return [*self.lanes(), *self.sparks()]

    def fabric_links(self) -> list[frozenset[str]]:
        """The cabled node sets from `SPARK_FABRIC_LINKS`; `[]` = unconstrained."""
        return _parse_fabric_links(self.spark_fabric_links)

    def fabric_link_ok(self, member_ids: list[str] | tuple[str, ...]) -> bool:
        """Can these nodes actually talk to each other?

        True when unconstrained, or when the set fits inside one cabled group.
        Subset rather than equality on purpose: a switched 4-node fabric declared
        as one group must still admit 2- and 3-node jobs.
        """
        links = self.fabric_links()
        if not links:
            return True
        want = set(member_ids)
        return any(want <= link for link in links)

    def fabric_links_describe(self) -> str:
        """Human-readable cabled sets, for error messages and UI notes."""
        return " / ".join("+".join(sorted(link)) for link in self.fabric_links()) or "unconstrained"

    def spark_group_config(self, member_ids: list[str] | tuple[str, ...]) -> SparkGroupConfig:
        """Build the config for a group over `member_ids` (order-insensitive).

        Raises ValueError on an unknown/disabled member, a set of fewer than two
        nodes, or a set that is not cabled together (`SPARK_FABRIC_LINKS`) — the
        callers (orchestrator startup + POST /api/cluster/load) surface that as a
        4xx rather than half-creating a unit. This is the single chokepoint for
        group creation, which is why the topology check belongs here and not in
        the REST layer: the startup re-instantiation of recorded node sets has to
        honour it too, or a set recorded before the cabling changed would come
        back as a standing auto-placement candidate.
        """
        ids = sorted(set(member_ids))
        if len(ids) < 2:
            raise ValueError(f"a spark group needs at least 2 members, got {ids}")
        by_id = {c.id: c for c in self.sparks()}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(
                f"unknown spark node(s) {missing} (have: {', '.join(by_id) or 'none — SPARK_ENABLED off?'})"
            )
        if not self.fabric_link_ok(ids):
            raise ValueError(
                f"{'+'.join(ids)} are not cabled together — a tensor-parallel job "
                f"across them would hang, not fail. Cabled sets: "
                f"{self.fabric_links_describe()}"
            )
        head = by_id[ids[0]]
        ordinals = [
            "".join(ch for ch in i if ch.isdigit()) or i
            for i in ids
        ]
        return SparkGroupConfig(
            id=group_id_for(ids),
            name=f"Sparks {'+'.join(ordinals)}",
            member_ids=tuple(ids),
            head_host=head.host,
            api_port=head.api_port,
            load_timeout_s=max(3600, self.spark_load_timeout_s),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
