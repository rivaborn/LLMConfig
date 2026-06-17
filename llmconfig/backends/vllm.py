"""vLLM backend — WSL2 server reached via the socat relay (:11437) for status,
and driven via `serve.sh` / `systemctl --user` over wsl.exe for lifecycle.

vLLM serves one model per process; "loading" a different model means restarting
the process with a new serve.sh alias. The relay's /v1/models reports the
currently-served name.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional

import httpx

from ..config import Settings
from ..registry import Registry
from ..schemas import VllmAlias
from ..wsl import run_wsl, user_journalctl, user_systemctl

LogCb = Callable[[str], None]


class VllmBackend:
    def __init__(self, settings: Settings, registry: Registry):
        self.s = settings
        self.registry = registry

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.s.vllm_relay_url,
            timeout=httpx.Timeout(timeout if timeout is not None else self.s.http_timeout_s),
        )

    def _unit(self, alias: str) -> str:
        return f"{self.s.vllm_systemd_unit}{alias}"  # e.g. "vllm@coder30-awq"

    # ---- liveness / state ----
    async def served(self) -> Optional[str]:
        """The currently-served model name (from the relay), or None if vLLM is down."""
        try:
            async with self._client() as c:
                r = await c.get("/v1/models")
                r.raise_for_status()
                data = r.json().get("data", []) or []
                return data[0].get("id") if data else None
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return None

    async def up(self) -> bool:
        return (await self.served()) is not None

    async def relay_up(self) -> bool:
        """True if the socat relay answers at all (even with nothing served)."""
        try:
            async with self._client() as c:
                r = await c.get("/v1/models")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_aliases(self) -> list[VllmAlias]:
        served = await self.served()
        out: list[VllmAlias] = []
        for entry in self.registry.entries():
            pub = entry.to_public()
            pub.loaded = bool(served and pub.served_name == served)
            out.append(pub)
        return out

    # ---- lifecycle ----
    async def serve(self, alias: str):
        """Stop any running vLLM instance, then (re)start the templated unit for `alias`."""
        await self.stop()
        return await run_wsl(
            user_systemctl(f"restart {self._unit(alias)}"),
            login=False,
            timeout=40.0,
            settings=self.s,
        )

    async def stop(self) -> None:
        # Stop any vllm@ instance, then belt-and-suspenders pkill the actual binary.
        await run_wsl(
            user_systemctl("stop 'vllm@*' 2>/dev/null; true"),
            login=False,
            timeout=30.0,
            settings=self.s,
        )
        await run_wsl(
            "pkill -f 'venv/bin/vllm' 2>/dev/null; true",
            login=False,
            timeout=15.0,
            settings=self.s,
        )

    async def wait_ready(
        self,
        served_name: str,
        timeout: float,
        on_log: LogCb | None = None,
        alias: str | None = None,
    ) -> bool:
        """Poll the relay until `served_name` is being served, or timeout."""
        deadline = time.monotonic() + timeout
        last_tail = ""
        while time.monotonic() < deadline:
            if (await self.served()) == served_name:
                return True
            if on_log and alias:
                tail = await self.journal_tail(alias, n=1)
                if tail and tail != last_tail:
                    last_tail = tail
                    on_log(tail)
            await asyncio.sleep(self.s.poll_interval_s)
        return False

    async def journal_tail(self, alias: str, n: int = 40) -> str:
        r = await run_wsl(
            user_journalctl(f"-u {self._unit(alias)} -n {n} --no-pager -o cat 2>/dev/null"),
            login=False,
            timeout=15.0,
            settings=self.s,
        )
        return r.out.strip()

    # ---- introspection (doctor / catalog refresh) ----
    async def serve_help(self):
        return await run_wsl(
            f"{self.s.vllm_serve_script} --help",
            login=True,
            timeout=20.0,
            settings=self.s,
        )

    async def unit_exists(self, alias: str = "smoke") -> bool:
        r = await run_wsl(
            user_systemctl(f"cat {self._unit(alias)} >/dev/null 2>&1 && echo yes || echo no"),
            login=False,
            timeout=15.0,
            settings=self.s,
        )
        return r.out.strip().endswith("yes")
