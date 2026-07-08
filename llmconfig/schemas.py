"""Pydantic models shared across the API, backends, and orchestrator."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ServerName = Literal["ollama", "vllm"]
Owner = Literal["free", "ollama", "vllm", "unknown"]
JobState = Literal["pending", "running", "succeeded", "failed"]


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


class ModelsResponse(BaseModel):
    ollama: list[OllamaModel] = Field(default_factory=list)
    vllm: list[VllmAlias] = Field(default_factory=list)
    ollama_error: str = ""
    vllm_error: str = ""


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
    utilization_pct: float = 0.0
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
            utilization_pct=g.utilization_pct,
            processes=[GpuProcessOut(pid=p.pid, used_mb=p.used_mb, name=p.name) for p in g.processes],
            error=g.error,
        )


# --------------------------------------------------------------------------- #
# Loaded model / status
# --------------------------------------------------------------------------- #
class LoadedModel(BaseModel):
    server: ServerName
    model: str
    size_bytes: int = 0
    on_gpu_bytes: int = 0
    on_cpu_bytes: int = 0
    spilled: bool = False
    fully_on_gpu: bool = True
    gpu_utilization_pct: float = 0.0


class LaneStatus(BaseModel):
    """Per-GPU lane state. The primary lane is the RTX 3090; an optional companion
    lane is the RTX 3070 Ti. Each lane independently arbitrates Ollama-XOR-vLLM."""

    id: str
    name: str
    enabled: bool = True
    owner: Owner
    ollama_up: bool
    vllm_up: bool
    loaded: Optional[LoadedModel] = None
    gpu: GpuOut
    swap_in_progress: bool = False
    active_job_id: Optional[str] = None
    idle_s: Optional[float] = None  # seconds since last observed activity (idle-reaper input)


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
