"""`llmconfig doctor` — read-only recon that verifies every on-box assumption.

Run it (against the live .40, locally or remotely over Tailscale) to confirm the
serve.sh path + alias map, the vLLM systemd unit, systemctl-over-wsl, service-
control rights, the 3090 UUID, and relay/Ollama reachability — i.e. everything
the plan flagged to confirm once the box is back.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from . import winsvc
from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import Settings, get_settings
from .gpu import query_gpu
from .registry import Registry, make_registry
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


async def run_doctor(settings: Settings | None = None, registry: Registry | None = None) -> DoctorReport:
    settings = settings or get_settings()
    registry = registry or make_registry(settings)
    ollama = OllamaBackend(settings)
    vllm = VllmBackend(settings, registry)
    checks: list[Check] = []

    def add(name: str, ok: Optional[bool], detail: str = "") -> None:
        checks.append(Check(name=name, ok=ok, detail=detail))

    # --- Ollama (Windows-native) ---
    ver = await ollama.version()
    if ver is not None:
        loaded = await ollama.loaded_names()
        add("ollama.api", True, f"v{ver} at {settings.ollama_url}; loaded: {', '.join(loaded) or 'none'}")
    else:
        add("ollama.api", False, f"no response at {settings.ollama_url}")

    st = await ollama.service_status()
    add("ollama.service", True if st == winsvc.RUNNING else None,
        f"status={st} (name={settings.ollama_service_name})")

    elevated = await winsvc.is_elevated()
    add("ollama.service_control", True if elevated else None,
        "running elevated - can Start/Restart-Service" if elevated
        else "NOT elevated - Start/Restart-Service may be access-denied; run the app as admin or grant the service ACL")

    # --- GPU ---
    gpu = await query_gpu(settings)
    if gpu.found:
        add("gpu.3090", True, f"{gpu.uuid}: {gpu.used_mb}/{gpu.total_mb} MiB used ({gpu.utilization_pct}%)")
    else:
        add("gpu.3090", False, gpu.error or "nvidia-smi did not report the configured UUID")

    # --- WSL plumbing ---
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

        r = await run_wsl(f"test -x {settings.vllm_serve_script} && echo ok || echo missing",
                          login=False, timeout=20.0, settings=settings)
        serve_present = "ok" in r.out
        add("vllm.serve_script", serve_present,
            settings.vllm_serve_script if serve_present else f"{settings.vllm_serve_script} not found/executable")

        if serve_present:
            help_res = await vllm.serve_help()
            help_text = help_res.out
            managed = [e.alias for e in registry.entries() if e.managed_by == "serve.sh"]
            missing = [a for a in managed if a not in help_text]
            if not help_text:
                add("vllm.aliases", None, f"serve.sh --help produced no output: {help_res.text()}")
            elif missing:
                add("vllm.aliases", None, f"registry aliases not in serve.sh --help: {', '.join(missing)}")
            else:
                add("vllm.aliases", True, f"all {len(managed)} registry aliases present in serve.sh --help")

        unit_ok = await vllm.unit_exists("smoke")
        add("vllm.systemd_unit", True if unit_ok else None,
            f"{settings.vllm_systemd_unit}<alias> template installed" if unit_ok
            else f"unit {settings.vllm_systemd_unit}<alias> not found — install deploy/vllm@.service")

    # --- vLLM relay ---
    relay = await vllm.relay_up()
    if relay:
        served = await vllm.served()
        add("vllm.relay", True, f"{settings.vllm_relay_url} up; serving: {served or 'nothing'}")
    else:
        add("vllm.relay", None, f"{settings.vllm_relay_url} not answering (expected when vLLM is stopped)")

    return DoctorReport(checks=checks).summarize()
