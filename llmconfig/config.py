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
    spark_stop_one_cmd: str = "sparkrun stop {recipe} --cluster {cluster} --hosts {host}"
    spark_status_cmd: str = "sparkrun status --cluster {cluster}"
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
    # Recent-activity window for classifying a loaded lane "active" (GET /api/usage
    # and the `usage` field on /api/status lanes).
    usage_active_window_s: float = 60.0

    # --- leases (resource sharing between callers — see leases.py) ---
    # A lease is a caller's claim on one unit: "I'm using this, and here's whether
    # you may take it from me." Advisory by design — clients that bypass LLMConfig
    # (direct Ollama on :11434/:11435) can't be stopped, so a non-preemptible lease
    # is a cooperation contract, not a hard exclusivity guarantee.
    lease_enabled: bool = True
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
        `sparks()` remain available for code that needs only one kind.
        """
        return [*self.lanes(), *self.sparks()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
