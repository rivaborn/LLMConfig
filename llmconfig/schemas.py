"""Pydantic models shared across the API, backends, and orchestrator."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr

ServerName = Literal["ollama", "vllm", "spark"]
Owner = Literal["free", "ollama", "vllm", "spark", "unknown"]
JobState = Literal["pending", "running", "succeeded", "failed"]
# free = nothing loaded; idle = model resident but unused; active = model in use
LaneUsage = Literal["free", "idle", "active"]
# "gpu"   = a local card arbitrated Ollama-XOR-vLLM (the 3090 / 3070 Ti lanes)
# "spark" = a remote DGX Spark node driven by sparkrun; the node IS the unit, so
#           there is no intra-unit arbitration and no local nvidia-smi.
UnitKind = Literal["gpu", "spark"]
# The set of owners that mean "something we manage holds this unit".
MANAGED_OWNERS: tuple[str, ...] = ("ollama", "vllm", "spark")
# Lease lifecycle. `released` = the holder handed it back; `revoked` = someone took
# it away (preemption or an operator). They stay distinct so a displaced holder can
# see *why* it lost the unit.
LeaseState = Literal["active", "released", "revoked", "expired"]


# --------------------------------------------------------------------------- #
# Models / catalog
# --------------------------------------------------------------------------- #
class OllamaModel(BaseModel):
    name: str
    server: ServerName = "ollama"
    size_bytes: int = 0
    modified: str = ""
    loaded: bool = False
    size_vram_bytes: int = 0  # portion on GPU when loaded (size_vram < size_bytes ⇒ spilled)
    context_len: int = 0      # /api/ps context_length — the RUNTIME window, 0 when not loaded


class ServedModel(BaseModel):
    """What an OpenAI-style backend reports it is currently serving.

    A record rather than a tuple so new fields (root, context) don't keep
    changing the arity every call site has to unpack.
    """

    name: Optional[str] = None
    root: str = ""           # the real HF repo behind `name` (names can collide across units)
    context_len: int = 0     # max_model_len as launched


class VllmAlias(BaseModel):
    alias: str
    server: ServerName = "vllm"
    hf_repo: str = ""
    served_name: str = ""
    mode: str = ""           # "compile" | "eager"
    status: str = "unknown"  # "ok" | "blocked" | "unverified" | ...
    notes: str = ""
    loaded: bool = False
    load_timeout_s: int = 240


class VllmAliasEntry(BaseModel):
    """Full registry record (persisted to vllm_models.yaml). Superset of VllmAlias."""

    alias: str
    hf_repo: str = ""
    served_name: str = ""
    mode: str = ""
    status: str = "ok"
    notes: str = ""
    launch_args: list[str] = Field(default_factory=list)
    load_timeout_s: int = 240
    # "serve.sh": launch by alias through serve.sh (keeps its tuned args + UUID pin).
    # "registry": custom alias launched directly from launch_args (for user-added models).
    managed_by: Literal["serve.sh", "registry"] = "serve.sh"

    def to_public(self) -> "VllmAlias":
        return VllmAlias(
            alias=self.alias,
            hf_repo=self.hf_repo,
            served_name=self.served_name or self.alias,
            mode=self.mode,
            status=self.status,
            notes=self.notes,
            load_timeout_s=self.load_timeout_s,
        )


class SparkModel(BaseModel):
    """Public view of one curated sparkrun recipe available on a Spark node."""

    alias: str
    server: ServerName = "spark"
    recipe: str = ""         # the sparkrun recipe id actually launched
    served_name: str = ""    # what the node's /v1/models reports once up
    tp: int = 1              # node count; 1 until the 200G switch lands
    status: str = "ok"       # "ok" | "blocked" | "unverified"
    notes: str = ""
    loaded: bool = False
    load_timeout_s: int = 900
    mem_fraction: float = 0.0  # declared share of the node pool; 0 = whole node
    port: int = 0              # the port it is CURRENTLY served on; 0 when not loaded


class SparkModelEntry(BaseModel):
    """Full curated catalog record (persisted to data/spark_models_<unit>.yaml).

    sparkrun can enumerate hundreds of registry recipes; this is the small
    hand-maintained set this lab actually serves on a given node — the same
    role `VllmAliasEntry` plays for vLLM.
    """

    alias: str
    recipe: str = ""
    served_name: str = ""
    tp: int = 1
    status: str = "ok"
    notes: str = ""
    extra_args: list[str] = Field(default_factory=list)
    load_timeout_s: int = 900
    # Share of the node's unified pool this model may claim, passed to sparkrun as
    # `--gpu-mem`. A Spark has no eviction-wait gate (nothing to wait for — memory is
    # released when the container stops), so this declared budget is the ONLY thing
    # that keeps co-resident models from colliding: the unit sums it over everything
    # already loaded and refuses a load that would exceed `spark_mem_headroom`.
    # 0.0 means "unset" — treated as a whole-node claim, which is the pre-multi-model
    # behaviour and keeps old catalogs working.
    mem_fraction: float = 0.0

    def to_public(self) -> "SparkModel":
        return SparkModel(
            alias=self.alias,
            recipe=self.recipe or self.alias,
            served_name=self.served_name or self.alias,
            tp=self.tp,
            status=self.status,
            notes=self.notes,
            load_timeout_s=self.load_timeout_s,
            mem_fraction=self.mem_fraction,
        )


class ModelsResponse(BaseModel):
    ollama: list[OllamaModel] = Field(default_factory=list)
    vllm: list[VllmAlias] = Field(default_factory=list)
    spark: list[SparkModel] = Field(default_factory=list)
    ollama_error: str = ""
    vllm_error: str = ""
    spark_error: str = ""


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
class GpuProcessOut(BaseModel):
    pid: int
    used_mb: int
    name: str


class GpuOut(BaseModel):
    found: bool
    uuid: str = ""
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    # Compute utilization (nvidia-smi utilization.gpu); None when unavailable.
    # Historically this field carried the VRAM fraction, which deadlocked external
    # idle gates (a resident model reads ~86% "busy" forever) — that value now
    # lives in vram_pct.
    utilization_pct: Optional[float] = None
    vram_pct: float = 0.0
    processes: list[GpuProcessOut] = Field(default_factory=list)
    error: str = ""

    @classmethod
    def from_info(cls, g) -> "GpuOut":
        return cls(
            found=g.found,
            uuid=g.uuid,
            total_mb=g.total_mb,
            used_mb=g.used_mb,
            free_mb=g.free_mb,
            utilization_pct=g.util_pct,
            vram_pct=g.vram_pct,
            processes=[GpuProcessOut(pid=p.pid, used_mb=p.used_mb, name=p.name) for p in g.processes],
            error=g.error,
        )


# --------------------------------------------------------------------------- #
# Loaded model / status
# --------------------------------------------------------------------------- #
class LoadedModel(BaseModel):
    server: ServerName
    model: str
    # The backend's own `root` — the actual HF repo behind the served name.
    # Served names are chosen per unit and CAN collide across units: the 3090
    # served `gemma-4-26b` from cyankiwi/…-AWQ-4bit at 32k ctx while the Sparks
    # served the same name from google/gemma-4-26B-A4B-it at 65k, and nothing in
    # the API distinguished them. Both backends already report `root` on
    # /v1/models; carrying it here makes the real artifact visible.
    root: str = ""
    # Context window the model is ACTUALLY being served at — vLLM/Spark
    # `max_model_len`, Ollama's per-run `context_length` from /api/ps. Not the
    # architectural maximum: Ollama truncates to OLLAMA_CONTEXT_LENGTH (4096 on
    # this box) and vLLM to whatever --max-model-len the launch set, so this is
    # the number a client's prompt budget must actually respect. 0 = unknown.
    context_len: int = 0
    size_bytes: int = 0
    on_gpu_bytes: int = 0
    on_cpu_bytes: int = 0
    spilled: bool = False
    fully_on_gpu: bool = True
    gpu_vram_pct: float = 0.0  # share of the card's VRAM in use once loaded
    # Port this model is served on. Only meaningful for a Spark, where several
    # models can be resident at once and each sparkrun workload gets its own port;
    # it is DISCOVERED by probing, never persisted, so it survives a restart. 0 on
    # a GPU lane, whose single occupant is always reached via the lane's own URL.
    port: int = 0


class LaneStatus(BaseModel):
    """Per-unit state. `kind="gpu"` lanes are local cards arbitrating Ollama-XOR-vLLM
    (primary = RTX 3090, companion = RTX 3070 Ti); `kind="spark"` units are remote
    DGX Spark nodes driven by sparkrun, where the node itself is the unit.

    The name is historical — it predates non-GPU units — but the shape is shared so
    every unit renders through one code path in the UI and one `/api/status` schema.
    """

    id: str
    name: str
    kind: UnitKind = "gpu"
    host: str = ""            # remote units: the node's LAN address; "" for local lanes
    reachable: bool = True    # remote units: last probe succeeded
    enabled: bool = True
    owner: Owner
    ollama_up: bool
    vllm_up: bool
    # The unit's primary resident model — `loaded_models[0]` when anything is loaded.
    # Kept scalar for backward compatibility: off-box consumers already switch on it
    # (see invariant 8), so multi-model support arrives as the ADDITIVE list below
    # rather than by changing this field's type.
    loaded: Optional[LoadedModel] = None
    # Every model resident on the unit. A GPU lane holds at most one (the
    # eviction-wait gate guarantees it); a Spark can hold several, each on its own
    # port. Always consistent with `loaded`: empty iff `loaded is None`.
    loaded_models: list[LoadedModel] = Field(default_factory=list)
    # Remote units have no local card, so this defaults to a not-found GpuOut rather
    # than being required; Spark telemetry (when available) is filled in from the
    # node's own nvidia-smi over SSH.
    gpu: GpuOut = Field(default_factory=lambda: GpuOut(found=False))
    swap_in_progress: bool = False
    active_job_id: Optional[str] = None
    idle_s: Optional[float] = None  # seconds since last observed activity (idle-reaper input)
    # free/idle/active classification — populated by the REST layer (it needs the
    # Monitor's current util, which the orchestrator doesn't hold); None in-process.
    usage: Optional[LaneUsage] = None
    # The live claim on this unit, if any. ADDITIVE — `usage` deliberately keeps its
    # three values because off-box consumers switch on it; a lease is never a
    # fourth usage state.
    lease: Optional["LeaseBrief"] = None


class StatusResponse(BaseModel):
    # Top-level fields mirror the PRIMARY lane (backward compatible); `lanes` carries
    # every lane (primary + companion).
    owner: Owner
    ollama_up: bool
    vllm_up: bool
    loaded: Optional[LoadedModel] = None
    gpu: GpuOut
    swap_in_progress: bool = False
    active_job_id: Optional[str] = None
    message: str = ""
    lanes: list[LaneStatus] = Field(default_factory=list)


class LaneUsageOut(BaseModel):
    """Compact per-lane usage answer for GET /api/usage."""

    lane: str
    state: LaneUsage
    model: Optional[str] = None   # None when free
    idle_s: Optional[float] = None
    lease: Optional["LeaseBrief"] = None   # additive; see LaneStatus.lease


class UsageResponse(BaseModel):
    # Top-level fields mirror the requested lane (default primary); `lanes` carries all.
    lane: str
    state: LaneUsage
    model: Optional[str] = None
    idle_s: Optional[float] = None
    lease: Optional["LeaseBrief"] = None   # additive; mirrors the requested lane
    lanes: list[LaneUsageOut] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class LoadRequest(BaseModel):
    server: ServerName
    model: str  # Ollama tag, or vLLM serve.sh alias
    lane: str = "primary"      # which GPU lane: "primary" (3090) | "companion" (3070 Ti)
    force: bool = False        # reload even if already the active model
    max_pack: bool = False     # push num_gpu to fill VRAM before spilling (Ollama)
    keep_alive: int = -1       # Ollama keep_alive; -1 = pin until swapped


class UnloadRequest(BaseModel):
    server: Optional[ServerName] = None  # None = free whatever holds the GPU
    lane: str = "primary"                # which GPU lane to free
    # Free ONE model, leaving any co-resident models on the unit running. Only a
    # Spark can have co-tenants; on a GPU lane this is either the single occupant or
    # a no-op. None (the default) keeps the original meaning — free the whole unit —
    # which is what the idle reaper and a lease's `free_on_preempt` still want.
    model: Optional[str] = None


# --------------------------------------------------------------------------- #
# Jobs (long load/unload/pull operations)
# --------------------------------------------------------------------------- #
class Job(BaseModel):
    id: str
    kind: str
    state: JobState = "pending"
    message: str = ""
    progress: Optional[float] = None  # 0..1 when known
    log: list[str] = Field(default_factory=list)
    result: Optional[dict] = None
    error: str = ""
    created_at: float = 0.0
    finished_at: Optional[float] = None


# --------------------------------------------------------------------------- #
# Leases (resource sharing — see leases.py)
# --------------------------------------------------------------------------- #
class Lease(BaseModel):
    """One caller's claim on one unit.

    Two distinct durations, deliberately: `expected_duration_s` is the honest
    "I need this for N hours" hint (recorded and displayed, never enforced), while
    `ttl_s` is a short renewable leash that must be refreshed to prove the holder
    is still alive. A long job can declare its real shape without a crashed client
    pinning the unit for hours.
    """

    id: str
    unit: str
    holder: str
    preemptible: bool = True
    priority: int = 0            # 0..100, clamped; only breaks ties among preemptible leases
    ttl_s: float                 # enforcement window — renew before it lapses
    expected_duration_s: float = 0.0   # honest hint; displayed, never enforced
    acquired_at: float
    expires_at: float            # wall clock, for display
    state: LeaseState = "active"
    # What the holder loaded / intends to load (advisory, recorded for humans).
    server: Optional[ServerName] = None
    model: str = ""
    note: str = ""
    # "Give me this unit with nothing loaded on it" — queues a deferred unload once
    # the claim is granted (an external CUDA job, Wait-GpuIdle). Default off: a load
    # already evicts the card itself, so freeing first is usually a wasted
    # drain/refill plus a window for a third party to grab the empty card.
    free_on_preempt: bool = False
    # Lifecycle bookkeeping
    renewed_at: Optional[float] = None
    renew_count: int = 0
    released_at: Optional[float] = None
    revoked_at: Optional[float] = None
    revoked_by: str = ""          # holder label of whoever took it
    revoked_lease_id: str = ""    # the lease that displaced this one
    revoked_reason: str = ""      # preempted | admin | expired | unused | server_restarted
    # Observability — the evidence trail behind "was my job kicked off?"
    last_seen_at: Optional[float] = None
    requests: int = 0

    # Monotonic deadline: authoritative for liveness, never serialized. Wall clock
    # is for display only — an NTP step must not mass-expire or mass-extend leases
    # (same reasoning as Lane._wait_vram_free using time.monotonic()).
    _deadline: float = PrivateAttr(default=0.0)

    def is_live(self, now_mono: float) -> bool:
        return self.state == "active" and now_mono < self._deadline

    def in_grace(self, now_mono: float, grace_s: float) -> bool:
        """Past the deadline but still inside the renew/allow window."""
        return self.state in ("active", "expired") and now_mono < self._deadline + grace_s


class LeaseBrief(BaseModel):
    """Compact projection carried on /api/status and /api/usage."""

    id: str
    holder: str
    preemptible: bool = True
    priority: int = 0
    expires_at: float = 0.0
    expires_in_s: float = 0.0
    model: str = ""


class LeaseClaimRequest(BaseModel):
    unit: str = "primary"
    holder: str
    preemptible: bool = True     # False = "must not be interrupted"
    priority: int = 0
    ttl_s: Optional[float] = None            # defaults from settings; clamped
    expected_duration_s: float = 0.0
    server: Optional[ServerName] = None
    model: str = ""
    note: str = ""
    force: bool = False          # steal an equal/higher-priority PREEMPTIBLE lease
    free_on_preempt: bool = False   # also evict whatever is loaded, once granted


class LeaseClaimResponse(BaseModel):
    lease: Lease
    displaced: Optional[LeaseBrief] = None
    # Set when the unit was already mid-generation as the claim landed. A
    # non-preemptible lease is a FORWARD guarantee only — work already running
    # keeps the unit until it finishes (there is no job cancellation).
    busy_with: Optional[dict] = None


class LeaseRenewRequest(BaseModel):
    ttl_s: Optional[float] = None


class LeaseRevokeRequest(BaseModel):
    reason: str = "admin"
    by: str = "operator"
    free: bool = False           # also queue a deferred unload of the unit


class LeaseListResponse(BaseModel):
    leases: list[Lease] = Field(default_factory=list)
    # Leases are in-memory: a restart drops them all. Clients use this to tell
    # "your lease was revoked" apart from "the server restarted".
    server_started_at: float = 0.0
