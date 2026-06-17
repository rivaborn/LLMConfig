"""Runtime configuration, loaded from environment / `.env` via pydantic-settings.

Every box-specific value lives here so the app can be retargeted at a different
host (or the live `.40` specifics confirmed via `llmconfig doctor`) without code
changes. Defaults match the documented `Alien-3070-TI` setup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the `llmconfig/` package dir. Used for default data paths.
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent


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

    # --- GPU ---
    gpu_uuid: str = "GPU-739bece9-8298-7993-f7dd-c8d86cb541f9"  # the RTX 3090
    vram_total_mb: int = 24576
    vram_free_baseline_mb: int = 1500  # "freed" / "maxed" threshold

    # --- HuggingFace (vLLM downloads) ---
    hf_token: str = ""

    # --- paths ---
    registry_path: Path = REPO_ROOT / "data" / "vllm_models.yaml"

    # --- timeouts / tuning (seconds) ---
    http_timeout_s: float = 10.0
    evict_timeout_s: float = 45.0
    poll_interval_s: float = 2.0
    default_vllm_load_timeout_s: int = 240

    @property
    def auth_enabled(self) -> bool:
        return bool(self.llmconfig_api_key.strip())

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.llmconfig_host in ("0.0.0.0", "") else self.llmconfig_host
        return f"http://{host}:{self.llmconfig_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
