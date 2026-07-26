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

Unlike a GPU `Lane` there is no eviction-wait gate — stopping a container releases
its memory outright, so there is nothing to poll for. The node can host SEVERAL
workloads at once, one per slot port (`api_port + N`); residency is discovered by
probing those ports, never persisted, so it survives a restart. Every failure path
degrades quietly (unreachable node → `found=False` / `None`) so a Spark that is
powered off never breaks `/api/status`.
"""
from __future__ import annotations

import asyncio
import base64
import shlex
import time
from typing import Callable, Optional

import httpx

from ..config import Settings, SparkConfig
from ..gpu import GPU_QUERY, METRICS_QUERY, GpuInfo, GpuMetric, _parse_float, _parse_int
from ..registry import SparkRegistry
from ..schemas import ServedModel, SparkModel
from ..wsl import run_wsl

LogCb = Callable[[str], None]

# One SSH round-trip for both halves of a Spark's telemetry. GB10 withholds every
# memory field from nvidia-smi ([N/A]) because the GPU shares the host's unified
# LPDDR5X pool, so /proc/meminfo is the only honest source for occupancy.
_STATS_SEP = "#MEM#"
# The separator MUST stay single-quoted: unquoted, `#` opens a shell comment, so
# the marker never prints *and* the grep after it on the same line never runs —
# which silently reduced this to a plain nvidia-smi call and left every Spark
# reporting 0 % memory. Pinned by test_stats_command_quotes_the_separator.
_MEMINFO = "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo 2>/dev/null"
_STATS_CMD = f"nvidia-smi {GPU_QUERY} 2>/dev/null; echo '{_STATS_SEP}'; {_MEMINFO}"
_METRICS_CMD = f"nvidia-smi {METRICS_QUERY} 2>/dev/null; echo '{_STATS_SEP}'; {_MEMINFO}"


class SparkBackend:
    def __init__(self, settings: Settings, cfg: SparkConfig, registry: SparkRegistry):
        self.s = settings
        self.cfg = cfg
        self.registry = registry
        # One pooled client per slot port — a node can serve several models at
        # once and each lives on its own port.
        self._clients: dict[int, httpx.AsyncClient] = {}

    # ---- HTTP plumbing ----
    def _client(self, port: Optional[int] = None) -> httpx.AsyncClient:
        """Pooled client for one slot. Keyed by port because a node can serve
        several models at once, each its own workload on its own port."""
        port = port or self.cfg.api_port
        c = self._clients.get(port)
        if c is None or c.is_closed:
            c = httpx.AsyncClient(
                base_url=self.cfg.api_base_for(port),
                timeout=httpx.Timeout(self.s.http_timeout_s),
            )
            self._clients[port] = c
        return c

    async def aclose(self) -> None:
        for c in list(self._clients.values()):
            if not c.is_closed:
                await c.aclose()
        self._clients.clear()

    # ---- liveness / state ----
    async def served(self) -> Optional[str]:
        """The model this node is currently serving, or None when nothing is up."""
        return (await self.served_info()).name

    async def served_info(self, port: Optional[int] = None) -> ServedModel:
        """What ONE slot is serving: name, real HF repo, and context window.

        Served names are chosen per unit and can collide across units, so the
        root is what tells two same-named models apart (see `LoadedModel.root`).
        A slot runs a single vLLM/SGLang process, so its /v1/models carries one
        entry — the multi-model dimension is across PORTS, not within a response.
        """
        try:
            r = await self._client(port).get(
                "/v1/models", timeout=self.s.vllm_probe_timeout_s
            )
            r.raise_for_status()
            data = r.json().get("data", []) or []
            if not data:
                return ServedModel()
            d = data[0]
            return ServedModel(
                name=d.get("id"),
                root=d.get("root") or "",
                context_len=int(d.get("max_model_len") or 0),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return ServedModel()

    async def served_slots(self) -> dict[int, ServedModel]:
        """Every occupied slot as {port: ServedModel}, probed CONCURRENTLY.

        This is how the port->model map is discovered instead of persisted: after
        an LLMConfig restart the resident models are found by asking the node.
        All probes run together so N slots cost one probe timeout, not N — status
        is on the UI's 2.5 s poll path (invariant 9).
        """
        ports = self.cfg.slot_ports
        results = await asyncio.gather(
            *(self.served_info(p) for p in ports), return_exceptions=True
        )
        out: dict[int, ServedModel] = {}
        for port, res in zip(ports, results):
            if isinstance(res, ServedModel) and res.name:
                out[port] = res
        return out

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
        """Catalog with residency. Several entries can be `loaded` at once, each
        carrying the port it is actually reachable on."""
        slots = await self.served_slots()
        by_name = {sm.name: port for port, sm in slots.items() if sm.name}
        out: list[SparkModel] = []
        for entry in self.registry.entries():
            pub = entry.to_public()
            port = by_name.get(pub.served_name)
            pub.loaded = port is not None
            pub.port = port or 0
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
                # Slot 0 unless the caller names a port — `kw` wins, so a load can
                # target the slot it allocated.
                **{"port": self.cfg.api_port, **kw},
            ).strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"bad sparkrun command template {template!r}: unknown placeholder {e}"
            ) from e

    async def run_recipe(self, recipe: str, tp: int = 1, extra: list[str] | None = None,
                         served: str = "", port: Optional[int] = None,
                         mem_fraction: float = 0.0, timeout: Optional[float] = None):
        """Launch a workload on this node. Returns the raw CmdResult.

        `served` pins `--served-model-name`, so the node reports exactly the name
        this app waits for and the /v1 resolver matches — rather than whatever the
        recipe happens to default to.

        `port` selects the slot; without it the workload lands on slot 0 and can
        only ever be the node's single occupant. `mem_fraction`, when set, becomes
        `--gpu-mem` — the declared budget that lets models coexist, since vLLM
        preallocates and a recipe's own default is typically 0.7-0.85 of the pool.
        """
        args = list(extra or [])
        if mem_fraction:
            args += ["--gpu-mem", str(mem_fraction)]
        cmd = self._fmt(
            self.s.spark_run_cmd,
            recipe=shlex.quote(recipe),
            tp=int(tp or 1),
            served=shlex.quote(served or recipe),
            extra=" ".join(args),
            **({"port": int(port)} if port else {}),
        )
        # Generous timeout: sparkrun pulls — or on some recipes BUILDS — the
        # Docker image before the container starts, and the per-recipe budget
        # knows that cost better than the node default (gemma declares 3600 s,
        # the node 900): capping at the node default timed real launches out.
        return await run_wsl(cmd, login=True,
                             timeout=float(timeout or self.cfg.load_timeout_s),
                             settings=self.s)

    async def stop(self, recipe: Optional[str] = None):
        """Stop workloads on this node.

        With `recipe`, stops just that one and leaves co-residents running — the
        whole point of multi-model. Without it, `--all` frees the entire node,
        which is still what the idle reaper and a lease's `free_on_preempt` want.
        """
        if recipe:
            cmd = self._fmt(self.s.spark_stop_one_cmd, recipe=shlex.quote(recipe))
        else:
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
        port: Optional[int] = None,
    ) -> bool:
        """Poll ONE slot's /v1/models until `served_name` appears, or timeout.

        Scoped to `port` because co-resident models are the normal case now: a
        neighbour on another slot is not evidence of the wrong state, so polling
        the node as a whole would let one model's readiness be satisfied — or
        contradicted — by another's.
        """
        deadline = time.monotonic() + timeout
        announced = False
        dead_checks = 0
        next_alive_probe = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            current = (await self.served_info(port)).name
            if current == served_name:
                return True
            if current and not announced and on_log:
                # This slot is serving something else — a genuine anomaly, unlike
                # a different model on a different port.
                on_log(
                    f"slot {port or self.cfg.api_port} is serving '{current}' "
                    f"(waiting for '{served_name}')"
                )
                announced = True
            # A server that failed at startup (bad budget, OOM) dies in under a
            # minute but leaves the port silent — polling HTTP alone burned the
            # entire budget (900 s for a 43 s failure, live on spark4
            # 2026-07-26). Every ~30 s, ask the node whether the exec'd process
            # still exists; two consecutive 'dead' answers end the wait. Two,
            # because a single probe can race the exec that starts the server.
            if time.monotonic() >= next_alive_probe:
                next_alive_probe = time.monotonic() + 30.0
                state = await self.serve_status(port or self.cfg.api_port)
                if state == "dead":
                    dead_checks += 1
                    if dead_checks >= 2:
                        if on_log:
                            on_log("server process died at startup — stopping the wait")
                        return False
                else:
                    dead_checks = 0
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

        **GB10 reports memory.total, memory.used AND memory.free all as `[N/A]`**
        (measured on the live cluster 2026-07-24 — a node serving a 26B model still
        printed `[N/A], [N/A], [N/A]`). The GPU shares the host's LPDDR5X pool, so
        the honest memory figure is the host's, not nvidia-smi's. One SSH round-trip
        fetches both and `/proc/meminfo` supplies whatever nvidia-smi withholds —
        otherwise a fully-loaded Spark reports 0 % forever.
        """
        r = await self._ssh(_STATS_CMD, timeout=20.0)
        if not r.ok:
            return GpuInfo(
                found=False,
                uuid=self.cfg.gpu_uuid,
                error=r.text() or f"{self.cfg.host}: nvidia-smi unreachable",
            )
        smi, mem_total, mem_used = self._split_stats(r.out)
        for line in smi:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            total = _parse_int(parts[1]) or mem_total or self.cfg.vram_total_mb
            used = _parse_int(parts[2]) or mem_used
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

    @staticmethod
    def _split_stats(out: str) -> tuple[list[str], int, int]:
        """Split the combined probe into (nvidia-smi rows, total_mb, used_mb).

        `used` is derived as MemTotal-MemAvailable, which on a dedicated inference
        node is the figure that answers "would another model fit" — the unified pool
        is shared between host and GPU, so there is no separate VRAM number to read.
        """
        smi_rows: list[str] = []
        total_kb = avail_kb = 0
        in_mem = False
        for line in out.splitlines():
            if line.strip() == _STATS_SEP:
                in_mem = True
                continue
            if not in_mem:
                if line.strip():
                    smi_rows.append(line)
            elif line.startswith("MemTotal:"):
                total_kb = _parse_int(line.split(":", 1)[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = _parse_int(line.split(":", 1)[1])
        total_mb = total_kb // 1024
        used_mb = max(0, (total_kb - avail_kb) // 1024) if total_kb else 0
        return smi_rows, total_mb, used_mb

    async def metrics(self) -> Optional["GpuMetric"]:
        """One telemetry sample for the Monitor tab, or None when unreachable.

        Same shape as a local card's `GpuMetric` so the Monitor stores and charts
        Sparks through the identical path; `index` is -1 because a remote node has
        no place in the local nvidia-smi ordering (and no NVAPI hotspot sensors).
        """
        r = await self._ssh(_METRICS_CMD, timeout=20.0)
        if not r.ok:
            return None
        smi, mem_total, mem_used = self._split_stats(r.out)
        for line in smi:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            # Same unified-memory fallback as gpu(): GB10 reports every memory
            # field as [N/A], so the host pool is the real occupancy signal.
            total = _parse_int(parts[6]) or mem_total or self.cfg.vram_total_mb
            used = _parse_int(parts[7]) or mem_used
            return GpuMetric(
                index=-1,
                uuid=self.cfg.gpu_uuid,
                name=self.cfg.name,
                temp_c=_parse_float(parts[3]),
                power_w=_parse_float(parts[4]),
                util_pct=_parse_float(parts[5]),
                mem_total_mb=total,
                mem_used_mb=used,
                mem_free_mb=_parse_int(parts[8]) or max(0, total - used),
            )
        return None

    # sparkrun's container is a `sleep infinity` placeholder; the actual server is
    # exec'd inside it via a script that logs to /tmp/sparkrun_serve.log IN the
    # container. `docker logs` therefore shows only the CUDA banner — reading it
    # made every startup failure look blank. These two helpers find the container
    # whose serve script mentions this slot's port and inspect the real state.
    def _serve_probe(self, port: int, tail: int = 0) -> str:
        tail_cmd = (f"tail -{int(tail)} /tmp/sparkrun_serve.log 2>/dev/null; "
                    if tail else "")
        # Base64 the whole script, exactly as sparkrun does for its own exec'd
        # commands. This probe crosses FOUR quoting layers (Python → Windows argv
        # → wsl.exe argv reconstruction → remote sh), and quotes do not survive
        # them: wsl.exe mangles embedded double quotes, and shlex.quote escapes
        # inner single quotes AS double-quote sequences ('"'"'). The first,
        # quoted, version of this probe degraded to matching every container, so
        # gemma's SERVE_ALIVE masked the dead embedder and the fast-fail never
        # fired (live, 2026-07-26). Base64's alphabet is transparent to all four.
        script = (
            "for c in $(docker ps -q); do "
            f"docker exec $c sh -c 'grep -q port.{int(port)} /tmp/sparkrun_serve.sh 2>/dev/null "
            "&& { kill -0 $(cat /tmp/sparkrun_serve.pid) 2>/dev/null && echo SERVE_ALIVE "
            f"|| echo SERVE_DEAD; {tail_cmd}" + "}' 2>/dev/null; done"
        )
        b64 = base64.b64encode(script.encode()).decode()
        return f"echo {b64} | base64 -d | sh"

    async def serve_status(self, port: int) -> str:
        """'alive' | 'dead' | 'unknown' for the slot's exec'd server process.

        SSH, so never on the status path — only the load path polls it, where a
        server that died at startup otherwise burns the whole wait_ready budget
        looking at a port that will never answer.
        """
        r = await self._ssh(self._serve_probe(port), timeout=20.0)
        out = r.out or ""
        if "SERVE_ALIVE" in out:
            return "alive"
        if "SERVE_DEAD" in out:
            return "dead"
        return "unknown"

    async def logs(self, n: int = 40, port: Optional[int] = None) -> str:
        """Best-effort diagnostics tail. With `port`, the slot's real serve log."""
        if port:
            r = await self._ssh(self._serve_probe(port, tail=n), timeout=30.0)
            out = (r.out or "").replace("SERVE_ALIVE", "").replace("SERVE_DEAD", "").strip()
            if out:
                return out
        r = await self._ssh(
            "docker ps --filter label=sparkrun --format '{{.Names}}' | head -1 | "
            "xargs -r docker logs --tail " + str(int(n)) + " 2>&1",
            timeout=30.0,
        )
        return (r.out or r.err).strip()
