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
    # Recent-activity window for classifying a loaded lane "active" (GET /api/usage
    # and the `usage` field on /api/status lanes).
    usage_active_window_s: float = 60.0

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
                )
            )
        return lanes


@lru_cache
def get_settings() -> Settings:
    return Settings()
