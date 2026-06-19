"""`llmconfig doctor` — read-only recon that verifies every on-box assumption.

Run it (against the live .40, locally or remotely over Tailscale) to confirm, per
GPU lane, the serve.sh path + alias map, the vLLM systemd unit, the Ollama API +
service, and the GPU UUID; plus the shared WSL plumbing (distro, systemctl --user,
lingering) and service-control rights.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from . import winsvc
from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import LaneConfig, Settings, get_settings
from .gpu import query_gpu
from .registry import DEFAULT_COMPANION_REGISTRY, Registry, make_registry
from .wsl import run_wsl, user_systemctl


class Check(BaseModel):
    name: str
    ok: Optional[bool]  # True=pass, False=fail, None=warn/info
    detail: str = ""


class DoctorReport(BaseModel):
    checks: list[Check]
    passed: int = 0
    failed: int = 0
    warnings: int = 0

    def summarize(self) -> "DoctorReport":
        self.passed = sum(1 for c in self.checks if c.ok is True)
        self.failed = sum(1 for c in self.checks if c.ok is False)
        self.warnings = sum(1 for c in self.checks if c.ok is None)
        return self


def _registry_for(settings: Settings, cfg: LaneConfig, primary: Registry | None) -> Registry:
    if cfg.id == "primary":
        return primary or make_registry(settings)
    return Registry(cfg.registry_path, default_path=DEFAULT_COMPANION_REGISTRY)


async def _check_lane(add, settings: Settings, cfg: LaneConfig, registry: Registry, wsl_ok: bool) -> None:
    """Per-lane checks, prefixed with the lane id (e.g. `companion.gpu`)."""
    ollama = OllamaBackend(settings, base_url=cfg.ollama_url, service_name=cfg.ollama_service_name)
    vllm = VllmBackend(
        settings,
        registry,
        relay_url=cfg.vllm_relay_url,
        serve_script=cfg.vllm_serve_script,
        systemd_unit=cfg.vllm_systemd_unit,
    )
    p = cfg.id

    # --- GPU (by UUID) ---
    gpu = await query_gpu(settings, uuid=cfg.gpu_uuid)
    if gpu.found:
        add(f"{p}.gpu", True, f"{cfg.name}: {gpu.uuid}: {gpu.used_mb}/{gpu.total_mb} MiB ({gpu.utilization_pct}%)")
    else:
        add(f"{p}.gpu", False, gpu.error or f"{cfg.gpu_uuid} not reported by nvidia-smi")

    # --- Ollama (Windows-native) ---
    ver = await ollama.version()
    if ver is not None:
        loaded = await ollama.loaded_names()
        add(f"{p}.ollama.api", True, f"v{ver} at {cfg.ollama_url}; loaded: {', '.join(loaded) or 'none'}")
    else:
        add(f"{p}.ollama.api", False, f"no response at {cfg.ollama_url}")

    st = await ollama.service_status()
    add(f"{p}.ollama.service", True if st == winsvc.RUNNING else None,
        f"status={st} (name={cfg.ollama_service_name})")

    # --- vLLM (per-lane serve script / unit / relay) ---
    if wsl_ok:
        r = await run_wsl(f"test -x {cfg.vllm_serve_script} && echo ok || echo missing",
                          login=False, timeout=20.0, settings=settings)
        serve_present = "ok" in r.out
        add(f"{p}.vllm.serve_script", serve_present,
            cfg.vllm_serve_script if serve_present else f"{cfg.vllm_serve_script} not found/executable")

        if serve_present:
            help_res = await vllm.serve_help()
            help_text = help_res.out
            managed = [e.alias for e in registry.entries() if e.managed_by == "serve.sh"]
            missing = [a for a in managed if a not in help_text]
            if not help_text:
                add(f"{p}.vllm.aliases", None, f"serve script --help produced no output: {help_res.text()}")
            elif missing:
                add(f"{p}.vllm.aliases", None, f"registry aliases not in serve --help: {', '.join(missing)}")
            else:
                add(f"{p}.vllm.aliases", True, f"all {len(managed)} registry aliases present in serve --help")

        unit_ok = await vllm.unit_exists("smoke")
        add(f"{p}.vllm.systemd_unit", True if unit_ok else None,
            f"{cfg.vllm_systemd_unit}<alias> template installed" if unit_ok
            else f"unit {cfg.vllm_systemd_unit}<alias> not found — install the lane's vllm unit")

    relay = await vllm.relay_up()
    if relay:
        served = await vllm.served()
        add(f"{p}.vllm.relay", True, f"{cfg.vllm_relay_url} up; serving: {served or 'nothing'}")
    else:
        add(f"{p}.vllm.relay", None, f"{cfg.vllm_relay_url} not answering (expected when vLLM is stopped)")

    # ad-hoc backends: release their pooled clients (doctor may run per API request)
    await ollama.aclose()
    await vllm.aclose()


async def run_doctor(settings: Settings | None = None, registry: Registry | None = None) -> DoctorReport:
    settings = settings or get_settings()
    lane_cfgs = settings.lanes()
    checks: list[Check] = []

    def add(name: str, ok: Optional[bool], detail: str = "") -> None:
        checks.append(Check(name=name, ok=ok, detail=detail))

    add("lanes", None, "running: " + ", ".join(f"{c.id} ({c.name})" for c in lane_cfgs))

    # --- distinct GPU UUIDs across lanes (a collision would cross-contend) ---
    uuids = [c.gpu_uuid for c in lane_cfgs]
    if len(set(uuids)) != len(uuids):
        add("lanes.distinct_gpus", False, f"two lanes share a GPU UUID: {uuids}")
    elif len(lane_cfgs) > 1:
        add("lanes.distinct_gpus", True, "each lane pins a distinct GPU UUID")

    # --- service-control rights (shared; needed to Start/Restart Ollama services) ---
    elevated = await winsvc.is_elevated()
    add("ollama.service_control", True if elevated else None,
        "running elevated - can Start/Restart-Service" if elevated
        else "NOT elevated - Start/Restart-Service may be access-denied; run the app as admin or grant the service ACL")

    # --- WSL plumbing (shared; one distro hosts every lane's vLLM relay) ---
    r = await run_wsl("echo ok", login=False, timeout=20.0, settings=settings)
    wsl_ok = r.ok and "ok" in r.out
    add("wsl.distro", wsl_ok, f"{settings.wsl_distro} as {settings.wsl_user}" if wsl_ok else (r.text() or "wsl.exe unavailable"))

    if wsl_ok:
        r = await run_wsl(user_systemctl("show-environment >/dev/null 2>&1 && echo ok || echo FAIL"),
                          login=False, timeout=20.0, settings=settings)
        sysd_ok = "ok" in r.out
        add("wsl.systemctl_user", sysd_ok,
            "systemctl --user works (XDG_RUNTIME_DIR resolved)" if sysd_ok
            else f"systemctl --user failed under wsl.exe: {r.text()}")

        r = await run_wsl("loginctl show-user \"$(whoami)\" -p Linger 2>/dev/null || true",
                          login=False, timeout=20.0, settings=settings)
        add("wsl.lingering", True if "Linger=yes" in r.out else None,
            r.out.strip() or "could not read linger state (loginctl)")

    # --- per-lane checks ---
    for cfg in lane_cfgs:
        await _check_lane(add, settings, cfg, _registry_for(settings, cfg, registry), wsl_ok)

    if not settings.companion_enabled:
        add("companion.enabled", None, "companion lane off (set COMPANION_ENABLED=1 to use the RTX 3070 Ti)")

    return DoctorReport(checks=checks).summarize()
