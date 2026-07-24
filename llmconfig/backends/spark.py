"""DGX Spark backend — a remote GB10 node driven by `sparkrun` from this box's WSL.

Three transports, each chosen because it is the most reliable source for its job:

* **Status** is read over **HTTP** from the node's own OpenAI endpoint
  (`http://<host>:<port>/v1/models`), exactly as `VllmBackend` reads the socat
  relay. Far more dependable than screen-scraping `sparkrun status`, and it
  yields the served model name directly.
* **Lifecycle** (start/stop a workload) goes through the **`sparkrun` CLI**, run
  via `wsl.exe … bash -lc` — sparkrun itself SSHes to the node and manages the
  Docker container. The command templates live in `Settings` because sparkrun's
  flags shift between releases; correct them in `.env`, not here.
* **Telemetry** is plain **SSH** to the node (`nvidia-smi`), reusing the CSV
  parsers in `gpu.py`. The control node's WSL already has passwordless key auth
  to every Spark.

Unlike a GPU `Lane` there is no eviction-wait gate: the node *is* the unit and
runs one workload at a time, so "swapping models" is stop-then-run. Every failure
path degrades quietly (unreachable node → `found=False` / `None`) so a Spark that
is powered off never breaks `/api/status`.
"""
from __future__ import annotations

import asyncio
import shlex
import time
from typing import Callable, Optional

import httpx

from ..config import Settings, SparkConfig
from ..gpu import GPU_QUERY, METRICS_QUERY, GpuInfo, GpuMetric, _parse_float, _parse_int
from ..registry import SparkRegistry
from ..schemas import SparkModel
from ..wsl import run_wsl

LogCb = Callable[[str], None]


class SparkBackend:
    def __init__(self, settings: Settings, cfg: SparkConfig, registry: SparkRegistry):
        self.s = settings
        self.cfg = cfg
        self.registry = registry
        self._http: httpx.AsyncClient | None = None

    # ---- HTTP plumbing ----
    def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.cfg.api_base,
                timeout=httpx.Timeout(self.s.http_timeout_s),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    # ---- liveness / state ----
    async def served(self) -> Optional[str]:
        """The model this node is currently serving, or None when nothing is up."""
        try:
            r = await self._client().get("/v1/models", timeout=self.s.vllm_probe_timeout_s)
            r.raise_for_status()
            data = r.json().get("data", []) or []
            return data[0].get("id") if data else None
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return None

    async def up(self) -> bool:
        return (await self.served()) is not None

    async def reachable(self) -> bool:
        """True when the node answers on the network at all (SSH probe).

        Distinguishes "node is off / unplugged" from "node is up but idle", which
        the UI renders differently (greyed-out vs free).
        """
        r = await self._ssh("true", timeout=10.0)
        return r.rc == 0

    async def list_models(self) -> list[SparkModel]:
        served = await self.served()
        out: list[SparkModel] = []
        for entry in self.registry.entries():
            pub = entry.to_public()
            pub.loaded = bool(served and pub.served_name == served)
            out.append(pub)
        return out

    # ---- lifecycle (sparkrun over WSL) ----
    def _fmt(self, template: str, **kw) -> str:
        """Render a sparkrun command template, tolerating unknown placeholders."""
        try:
            return template.format(
                cluster=self.s.spark_cluster,
                host=self.cfg.host,
                user=self.cfg.ssh_user,
                **kw,
            ).strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"bad sparkrun command template {template!r}: unknown placeholder {e}"
            ) from e

    async def run_recipe(self, recipe: str, tp: int = 1, extra: list[str] | None = None):
        """Launch a workload on this node. Returns the raw CmdResult."""
        cmd = self._fmt(
            self.s.spark_run_cmd,
            recipe=shlex.quote(recipe),
            tp=int(tp or 1),
            extra=" ".join(extra or []),
        )
        # Generous timeout: sparkrun pulls the image/weights on a cold node.
        return await run_wsl(cmd, login=True, timeout=float(self.cfg.load_timeout_s), settings=self.s)

    async def stop(self):
        cmd = self._fmt(self.s.spark_stop_cmd)
        return await run_wsl(cmd, login=True, timeout=120.0, settings=self.s)

    async def cluster_status(self):
        cmd = self._fmt(self.s.spark_status_cmd)
        return await run_wsl(cmd, login=True, timeout=60.0, settings=self.s)

    async def wait_ready(
        self,
        served_name: str,
        timeout: float,
        on_log: LogCb | None = None,
    ) -> bool:
        """Poll the node's /v1/models until `served_name` appears, or timeout."""
        deadline = time.monotonic() + timeout
        announced = False
        while time.monotonic() < deadline:
            current = await self.served()
            if current == served_name:
                return True
            if current and not announced and on_log:
                # Something else came up — surface it rather than silently timing out.
                on_log(f"node is serving '{current}' (waiting for '{served_name}')")
                announced = True
            await asyncio.sleep(self.s.poll_interval_s)
        return False

    # ---- remote telemetry (SSH → nvidia-smi, parsed by gpu.py) ----
    async def _ssh(self, remote_command: str, timeout: float = 20.0):
        cmd = self.s.spark_ssh_cmd.format(
            user=self.cfg.ssh_user,
            host=self.cfg.host,
            command=shlex.quote(remote_command),
        )
        return await run_wsl(cmd, login=False, timeout=timeout, settings=self.s)

    async def gpu(self) -> GpuInfo:
        """This node's GPU as a `GpuInfo`, so Spark units render like GPU lanes.

        GB10 is unified memory and nvidia-smi reports `memory.total` as `[N/A]`,
        so a zero total falls back to the configured pool size — otherwise every
        Spark would show 0 % VRAM forever.
        """
        r = await self._ssh(f"nvidia-smi {GPU_QUERY}", timeout=20.0)
        if not r.ok:
            return GpuInfo(
                found=False,
                uuid=self.cfg.gpu_uuid,
                error=r.text() or f"{self.cfg.host}: nvidia-smi unreachable",
            )
        for line in r.out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            total = _parse_int(parts[1]) or self.cfg.vram_total_mb
            used = _parse_int(parts[2])
            free = _parse_int(parts[3]) or max(0, total - used)
            return GpuInfo(
                found=True,
                uuid=self.cfg.gpu_uuid,
                total_mb=total,
                used_mb=used,
                free_mb=free,
                util_pct=_parse_float(parts[4]) if len(parts) > 4 else None,
            )
        return GpuInfo(found=False, uuid=self.cfg.gpu_uuid, error="nvidia-smi returned no GPU rows")

    async def metrics(self) -> Optional["GpuMetric"]:
        """One telemetry sample for the Monitor tab, or None when unreachable.

        Same shape as a local card's `GpuMetric` so the Monitor stores and charts
        Sparks through the identical path; `index` is -1 because a remote node has
        no place in the local nvidia-smi ordering (and no NVAPI hotspot sensors).
        """
        r = await self._ssh(f"nvidia-smi {METRICS_QUERY}", timeout=20.0)
        if not r.ok:
            return None
        for line in r.out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            return GpuMetric(
                index=-1,
                uuid=self.cfg.gpu_uuid,
                name=self.cfg.name,
                temp_c=_parse_float(parts[3]),
                power_w=_parse_float(parts[4]),
                util_pct=_parse_float(parts[5]),
                mem_total_mb=_parse_int(parts[6]) or self.cfg.vram_total_mb,
                mem_used_mb=_parse_int(parts[7]),
                mem_free_mb=_parse_int(parts[8]),
            )
        return None

    async def logs(self, n: int = 40) -> str:
        """Best-effort tail of the node's running container, for job diagnostics."""
        r = await self._ssh(
            "docker ps --filter label=sparkrun --format '{{.Names}}' | head -1 | "
            "xargs -r docker logs --tail " + str(int(n)) + " 2>&1",
            timeout=30.0,
        )
        return (r.out or r.err).strip()
