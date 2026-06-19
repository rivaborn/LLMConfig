"""Ollama backend — Windows-native server at :11434 + its Windows service.

Loads/unloads are driven entirely through the REST API (`keep_alive` controls
residency); `size_vram < size` in /api/ps is how we detect CPU spill.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import httpx

from .. import winsvc
from ..config import Settings
from ..schemas import OllamaModel


class OllamaBackend:
    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        service_name: str | None = None,
    ):
        self.s = settings
        self.base_url = base_url or settings.ollama_url
        self.service_name = service_name or settings.ollama_service_name
        self._http: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        """A long-lived, connection-pooling client reused across calls (so repeated
        /api/ps, /api/version, etc. don't pay TCP setup every time). Per-request
        timeouts are passed at the call site; the client default is http_timeout_s."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.s.http_timeout_s),
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    # ---- liveness ----
    async def version(self) -> Optional[str]:
        try:
            r = await self._client().get("/api/version")
            r.raise_for_status()
            return r.json().get("version", "")
        except httpx.HTTPError:
            return None

    async def up(self) -> bool:
        return (await self.version()) is not None

    # ---- catalog / state ----
    async def _ps_raw(self) -> list[dict]:
        try:
            r = await self._client().get("/api/ps")
            r.raise_for_status()
            return r.json().get("models", []) or []
        except httpx.HTTPError:
            return []

    async def list_models(self) -> list[OllamaModel]:
        r = await self._client().get("/api/tags")
        r.raise_for_status()
        tags = r.json().get("models", []) or []
        loaded = {m.get("name", ""): m for m in await self._ps_raw()}
        out: list[OllamaModel] = []
        for m in tags:
            name = m.get("name", "")
            ps = loaded.get(name)
            out.append(
                OllamaModel(
                    name=name,
                    size_bytes=int(m.get("size", 0) or 0),
                    modified=str(m.get("modified_at", "") or ""),
                    loaded=ps is not None,
                    size_vram_bytes=int((ps or {}).get("size_vram", 0) or 0),
                )
            )
        return out

    async def loaded(self) -> list[OllamaModel]:
        out: list[OllamaModel] = []
        for m in await self._ps_raw():
            out.append(
                OllamaModel(
                    name=m.get("name", ""),
                    size_bytes=int(m.get("size", 0) or 0),
                    loaded=True,
                    size_vram_bytes=int(m.get("size_vram", 0) or 0),
                )
            )
        return out

    async def loaded_names(self) -> list[str]:
        return [m.get("name", "") for m in await self._ps_raw() if m.get("name")]

    # ---- load / unload ----
    async def load(
        self,
        model: str,
        *,
        keep_alive: int = -1,
        num_gpu: int | None = None,
        timeout: float = 900.0,
    ) -> dict:
        """Load (and pin) a model into memory. Empty prompt = load-only, returns when ready."""
        body: dict = {"model": model, "keep_alive": keep_alive, "stream": False}
        if num_gpu is not None:
            body["options"] = {"num_gpu": num_gpu}
        r = await self._client().post("/api/generate", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    async def unload(self, model: str) -> None:
        r = await self._client().post(
            "/api/generate", json={"model": model, "keep_alive": 0, "stream": False}
        )
        r.raise_for_status()

    async def unload_all(self) -> list[str]:
        names = await self.loaded_names()
        for n in names:
            try:
                await self.unload(n)
            except httpx.HTTPError:
                pass
        return names

    # ---- service control ----
    async def service_status(self) -> str:
        return await winsvc.service_status(self.service_name)

    async def ensure_running(self, wait_s: float = 20.0) -> bool:
        if await self.up():
            return True
        st = await self.service_status()
        if st != winsvc.RUNNING:
            await winsvc.start_service(self.service_name)
        # poll the API rather than trusting the service state alone
        import asyncio
        import time

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if await self.up():
                return True
            await asyncio.sleep(1.0)
        return await self.up()

    async def restart(self) -> bool:
        r = await winsvc.restart_service(self.service_name)
        return r.ok

    # ---- model management ----
    async def pull(self, model: str, on_event: Callable[[dict], None] | None = None) -> None:
        # Dedicated client: pulls stream for minutes (timeout=None), so they don't share
        # the pooled client's default timeout.
        async with httpx.AsyncClient(base_url=self.base_url, timeout=None) as c:
            async with c.stream("POST", "/api/pull", json={"model": model, "stream": True}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if on_event:
                        on_event(evt)
                    if evt.get("error"):
                        raise RuntimeError(evt["error"])

    async def delete(self, model: str) -> None:
        r = await self._client().request("DELETE", "/api/delete", json={"model": model})
        r.raise_for_status()

    async def show(self, model: str) -> dict:
        r = await self._client().post("/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    async def block_count(self, model: str) -> int:
        """Total transformer layers for the model (for max-pack num_gpu math). 0 if unknown."""
        try:
            info = await self.show(model)
        except httpx.HTTPError:
            return 0
        model_info = info.get("model_info", {}) or {}
        for key, val in model_info.items():
            if key.endswith(".block_count"):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return 0
        return 0
