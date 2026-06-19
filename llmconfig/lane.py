"""One GPU lane — the arbitration state machine for a single card.

Each `Lane` pins one GPU (by UUID) and guarantees the requested model becomes the
*sole* occupant of that card: the other server plus any other Ollama models are
evicted and the VRAM is confirmed freed (via nvidia-smi) **before** the target is
loaded, so it packs 100 % of VRAM before any CPU spill. All swaps on a lane are
serialized behind that lane's own lock.

The `Orchestrator` runs one Lane per `LaneConfig` (primary = RTX 3090, optional
companion = RTX 3070 Ti); the lanes are fully independent — loading on one never
touches the other's card.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from .backends.ollama import OllamaBackend
from .backends.vllm import VllmBackend
from .config import LaneConfig, Settings
from .gpu import GpuInfo, query_gpu
from .jobs import JobManager
from .registry import Registry
from .schemas import Job, LaneStatus, LoadedModel, LoadRequest, UnloadRequest
from .schemas import GpuOut
from .wsl import WslKeepalive


class Lane:
    def __init__(
        self,
        settings: Settings,
        cfg: LaneConfig,
        registry: Registry,
        jobs: JobManager,
        keepalive: WslKeepalive,
    ):
        self.s = settings
        self.cfg = cfg
        self.registry = registry
        self.jobs = jobs
        self.keepalive = keepalive
        self.ollama = OllamaBackend(
            settings, base_url=cfg.ollama_url, service_name=cfg.ollama_service_name
        )
        self.vllm = VllmBackend(
            settings,
            registry,
            relay_url=cfg.vllm_relay_url,
            serve_script=cfg.vllm_serve_script,
            systemd_unit=cfg.vllm_systemd_unit,
        )
        self._lock = asyncio.Lock()
        self._active_job_id: Optional[str] = None

    async def _gpu(self) -> GpuInfo:
        """This lane's card only (by UUID) — never the other lane's."""
        return await query_gpu(self.s, uuid=self.cfg.gpu_uuid)

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    async def status(self, gpu: GpuInfo | None = None) -> LaneStatus:
        # `gpu` may be supplied by the Orchestrator (one nvidia-smi shared across
        # lanes); fetch this lane's card only when called standalone.
        if gpu is None:
            served, ollama_loaded, ollama_up, gpu = await asyncio.gather(
                self.vllm.served(),
                self.ollama.loaded(),
                self.ollama.up(),
                self._gpu(),
            )
        else:
            served, ollama_loaded, ollama_up = await asyncio.gather(
                self.vllm.served(),
                self.ollama.loaded(),
                self.ollama.up(),
            )

        loaded: Optional[LoadedModel] = None
        if served:
            owner = "vllm"
            loaded = LoadedModel(
                server="vllm",
                model=served,
                gpu_utilization_pct=gpu.utilization_pct,
                fully_on_gpu=True,
            )
        elif ollama_loaded:
            owner = "ollama"
            m = max(ollama_loaded, key=lambda x: x.size_vram_bytes)
            on_cpu = max(0, m.size_bytes - m.size_vram_bytes)
            loaded = LoadedModel(
                server="ollama",
                model=m.name,
                size_bytes=m.size_bytes,
                on_gpu_bytes=m.size_vram_bytes,
                on_cpu_bytes=on_cpu,
                spilled=on_cpu > 0,
                fully_on_gpu=on_cpu == 0,
                gpu_utilization_pct=gpu.utilization_pct,
            )
        else:
            owner = "free" if (not gpu.found or gpu.is_free(self.cfg.vram_free_baseline_mb)) else "unknown"

        return LaneStatus(
            id=self.cfg.id,
            name=self.cfg.name,
            enabled=self.cfg.enabled,
            owner=owner,
            ollama_up=ollama_up,
            vllm_up=served is not None,
            loaded=loaded,
            gpu=GpuOut.from_info(gpu),
            swap_in_progress=self._lock.locked(),
            active_job_id=self._active_job_id,
        )

    # ------------------------------------------------------------------ #
    # Load (returns a Job; the swap runs in the background under the lock)
    # ------------------------------------------------------------------ #
    def load(self, req: LoadRequest) -> Job:
        job = self.jobs.create(kind=f"load:{self.cfg.id}:{req.server}:{req.model}")

        async def body(job: Job) -> dict:
            if self._lock.locked():
                self.jobs.log(job, "waiting for an in-progress swap to finish…")
            async with self._lock:
                self._active_job_id = job.id
                try:
                    if req.server == "ollama":
                        return await self._load_ollama(job, req)
                    return await self._load_vllm(job, req)
                finally:
                    self._active_job_id = None

        return self.jobs.start(job, body)

    async def _load_ollama(self, job: Job, req: LoadRequest) -> dict:
        # Fast path: already loaded, nothing else on the GPU, not forced.
        if not req.force and (await self.vllm.served()) is None:
            if req.model in await self.ollama.loaded_names():
                self.jobs.log(job, f"{req.model} already loaded on Ollama")
                return await self._verify_ollama(job, req, remediate=False)

        await self._evict_all(job)

        self.jobs.log(job, "ensuring Ollama service is running…")
        if not await self.ollama.ensure_running():
            raise RuntimeError("Ollama service is not reachable (check the Windows service / OLLAMA_URL)")

        num_gpu = None  # default: let Ollama auto-fit against the now-empty GPU
        self.jobs.log(job, f"loading {req.model} into Ollama…")
        await self.ollama.load(req.model, keep_alive=req.keep_alive, num_gpu=num_gpu)
        return await self._verify_ollama(job, req, remediate=req.max_pack)

    async def _load_vllm(self, job: Job, req: LoadRequest) -> dict:
        alias = req.model
        entry = self.registry.get(alias)
        if entry is None:
            raise RuntimeError(f"unknown vLLM alias '{alias}' (see GET /api/models)")
        if entry.status == "blocked" and not req.force:
            raise RuntimeError(f"alias '{alias}' is blocked: {entry.notes}. Re-issue with force=true to try anyway.")

        served_target = entry.served_name or alias
        if not req.force and (await self.vllm.served()) == served_target:
            self.jobs.log(job, f"vLLM already serving {served_target}")
            return self._vllm_result(served_target, await self._gpu())

        # Hold WSL open before starting vLLM: otherwise the distro idle-shuts-down
        # seconds after this load returns and takes the model (and relay) with it.
        if not self.keepalive.ensure():
            self.jobs.log(job, "warning: could not start the WSL keepalive (wsl.exe missing?); "
                               "vLLM may not survive WSL idle-shutdown")
        else:
            self.jobs.log(job, "WSL keepalive active (distro held open)")

        self.jobs.log(job, "unloading any Ollama models…")
        names = await self.ollama.unload_all()
        if names:
            self.jobs.log(job, f"unloaded Ollama: {', '.join(names)}")
        await self._wait_vram_free(job)

        self.jobs.log(job, f"starting vLLM: serve.sh {alias}…")
        r = await self.vllm.serve(alias)
        if not r.ok and ("not found" in r.err.lower() or "not loaded" in r.err.lower()):
            raise RuntimeError(
                f"systemd unit '{self.cfg.vllm_systemd_unit}{alias}' not found — install deploy/vllm@.service "
                f"into ~/.config/systemd/user/ and `systemctl --user daemon-reload`. Detail: {r.text()}"
            )

        timeout = float(entry.load_timeout_s or self.s.default_vllm_load_timeout_s)
        self.jobs.log(job, f"waiting up to {int(timeout)}s for {served_target} to be ready…")
        ok = await self.vllm.wait_ready(
            served_target, timeout, on_log=lambda l: self.jobs.log(job, l), alias=alias
        )
        if not ok:
            # A heavy `mode: compile` alias can report ready a beat after its per-alias
            # deadline; re-check briefly before failing, so we don't fail — and have the
            # load torn down downstream — a vLLM that actually came up.
            self.jobs.log(job, f"readiness wait hit {int(timeout)}s; grace re-check ({self.s.vllm_ready_grace_s}s)…")
            ok = await self.vllm.wait_ready(served_target, float(self.s.vllm_ready_grace_s), alias=alias)
        if not ok:
            tail = await self.vllm.journal_tail(alias, n=25)
            raise RuntimeError(
                f"vLLM did not become ready for '{alias}' within {int(timeout)}s "
                f"(+{self.s.vllm_ready_grace_s}s grace).\n{tail}"
            )

        gpu = await self._gpu()
        self.jobs.log(job, f"vLLM serving {served_target} (GPU {gpu.utilization_pct}% used)")
        return self._vllm_result(served_target, gpu)

    # ------------------------------------------------------------------ #
    # Unload (synchronous eviction)
    # ------------------------------------------------------------------ #
    async def unload(self, req: UnloadRequest) -> LaneStatus:
        async with self._lock:
            self._active_job_id = None
            if req.server in (None, "vllm"):
                if await self.vllm.up():
                    await self.vllm.stop()
            if req.server in (None, "ollama"):
                await self.ollama.unload_all()
            await self._wait_vram_free(None)
        return await self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _evict_all(self, job: Job) -> None:
        """Clear this lane's GPU: stop vLLM, unload all Ollama models, confirm freed."""
        if await self.vllm.up():
            self.jobs.log(job, "stopping vLLM to free VRAM…")
            await self.vllm.stop()
        names = await self.ollama.unload_all()
        if names:
            self.jobs.log(job, f"unloaded Ollama: {', '.join(names)}")
        await self._wait_vram_free(job)

    async def _wait_vram_free(self, job: Optional[Job]) -> bool:
        """Block until this lane's card is back to driver baseline (the 100%-VRAM gate).

        If nvidia-smi can't see the card (off-box), don't block — return True.
        """
        deadline = time.monotonic() + self.s.evict_timeout_s
        while time.monotonic() < deadline:
            gpu = await self._gpu()
            if not gpu.found:
                if job:
                    self.jobs.log(job, "nvidia-smi unavailable — skipping VRAM-free wait")
                return True
            if gpu.is_free(self.cfg.vram_free_baseline_mb):
                if job:
                    self.jobs.log(job, f"VRAM free ({gpu.used_mb} MiB used)")
                return True
            if job:
                self.jobs.log(job, f"waiting for VRAM to free… ({gpu.used_mb} MiB still used)")
            await asyncio.sleep(self.s.poll_interval_s)
        if job:
            self.jobs.log(job, "warning: VRAM did not return to baseline before timeout; continuing")
        return False

    async def _verify_ollama(self, job: Job, req: LoadRequest, remediate: bool) -> dict:
        await asyncio.sleep(0.5)
        gpu = await self._gpu()
        match = next((m for m in await self.ollama.loaded() if m.name == req.model), None)
        if match is None:
            raise RuntimeError(f"{req.model} is not loaded after the request (check Ollama logs)")

        on_cpu = max(0, match.size_bytes - match.size_vram_bytes)
        spilled = on_cpu > 0
        # "Premature spill": spilled while the card still has substantial free VRAM.
        premature = spilled and gpu.found and gpu.free_mb > 2 * self.cfg.vram_free_baseline_mb

        if premature and remediate:
            self.jobs.log(job, f"premature spill detected ({gpu.free_mb} MiB free) — attempting max-pack reload")
            packed = await self._max_pack_reload(job, req, gpu)
            if packed is not None:
                return packed
            # fall through to report the original load

        self.jobs.log(
            job,
            f"loaded {req.model}: {_gib(match.size_vram_bytes)} on GPU / "
            f"{_gib(on_cpu)} on CPU; GPU {gpu.utilization_pct}% used"
            + (" — WARNING premature spill" if premature else ""),
        )
        return LoadedModel(
            server="ollama",
            model=req.model,
            size_bytes=match.size_bytes,
            on_gpu_bytes=match.size_vram_bytes,
            on_cpu_bytes=on_cpu,
            spilled=spilled,
            fully_on_gpu=not spilled,
            gpu_utilization_pct=gpu.utilization_pct,
        ).model_dump()

    async def _max_pack_reload(self, job: Job, req: LoadRequest, gpu: GpuInfo) -> Optional[dict]:
        """Best-effort: force num_gpu to fill VRAM, then reload once. Falls back on OOM."""
        layers = await self.ollama.block_count(req.model)
        match = next((m for m in await self.ollama.loaded() if m.name == req.model), None)
        if not layers or match is None or match.size_bytes <= 0:
            self.jobs.log(job, "max-pack: layer count unknown; leaving Ollama auto-fit result")
            return None

        usable_mb = max(0, self.cfg.vram_total_mb - self.cfg.vram_free_baseline_mb)
        per_layer_bytes = match.size_bytes / layers
        target_layers = int((usable_mb * 1024 * 1024 * 0.9) / per_layer_bytes)
        target_layers = max(1, min(target_layers, layers))
        self.jobs.log(job, f"max-pack: reloading with num_gpu={target_layers}/{layers}")
        try:
            await self.ollama.unload(req.model)
            await self._wait_vram_free(job)
            await self.ollama.load(req.model, keep_alive=req.keep_alive, num_gpu=target_layers)
        except Exception as e:  # OOM or similar — recover with the plain auto-fit load
            self.jobs.log(job, f"max-pack reload failed ({e}); restoring auto-fit load")
            try:
                await self.ollama.load(req.model, keep_alive=req.keep_alive)
            except Exception:
                pass
            return None
        return await self._verify_ollama(job, req, remediate=False)

    def _vllm_result(self, served_name: str, gpu: GpuInfo) -> dict:
        return LoadedModel(
            server="vllm",
            model=served_name,
            gpu_utilization_pct=gpu.utilization_pct,
            fully_on_gpu=True,
        ).model_dump()


def _gib(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GiB"
