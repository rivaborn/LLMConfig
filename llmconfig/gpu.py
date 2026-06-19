"""GPU truth via nvidia-smi — the arbitration signal.

Identifies the RTX 3090 strictly by **UUID** (the chassis 3070 Ti flaps in/out of
CUDA enumeration, so indices are unstable). Tries Windows `nvidia-smi` first, then
falls back to `nvidia-smi` inside WSL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .proc import CmdResult, run_argv
from .wsl import run_wsl

GPU_QUERY = "--query-gpu=uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
APPS_QUERY = "--query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits"


@dataclass
class GpuProcess:
    pid: int
    used_mb: int
    name: str


@dataclass
class GpuInfo:
    found: bool
    uuid: str = ""
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    processes: list[GpuProcess] = field(default_factory=list)
    error: str = ""

    @property
    def utilization_pct(self) -> float:
        return round(100.0 * self.used_mb / self.total_mb, 1) if self.total_mb else 0.0

    def is_free(self, baseline_mb: int) -> bool:
        """True when the card holds nothing but driver baseline (eviction complete)."""
        return self.found and self.used_mb <= baseline_mb


async def _run_smi(query: str, settings: Settings) -> CmdResult:
    """Run an nvidia-smi query string on Windows, falling back into WSL."""
    r = await run_argv(["nvidia-smi", *query.split()], timeout=15.0)
    if r.rc == 127:  # not on the Windows PATH — try inside WSL
        r = await run_wsl(f"nvidia-smi {query}", login=False, timeout=20.0, settings=settings)
    return r


def _parse_int(s: str) -> int:
    try:
        return int(s.strip().split()[0])
    except (ValueError, IndexError):
        return 0


async def query_gpu(settings: Settings | None = None, uuid: str | None = None) -> GpuInfo:
    """Query one GPU by UUID. Defaults to the primary card (`settings.gpu_uuid`);
    pass `uuid` to target another lane's card (e.g. the 3070 Ti companion)."""
    settings = settings or get_settings()
    target = uuid or settings.gpu_uuid
    r = await _run_smi(GPU_QUERY, settings)
    if not r.ok:
        return GpuInfo(found=False, error=r.text() or "nvidia-smi failed")

    seen: list[str] = []
    for line in r.out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        u, total, used, free = parts[0], _parse_int(parts[1]), _parse_int(parts[2]), _parse_int(parts[3])
        seen.append(u)
        if u == target:
            info = GpuInfo(found=True, uuid=u, total_mb=total, used_mb=used, free_mb=free)
            info.processes = await _query_processes(settings)
            return info

    return GpuInfo(
        found=False,
        error=f"GPU {target} not present. Visible UUIDs: {', '.join(seen) or 'none'}",
    )


async def _query_processes(settings: Settings) -> list[GpuProcess]:
    r = await _run_smi(APPS_QUERY, settings)
    if not r.ok:
        return []
    procs: list[GpuProcess] = []
    for line in r.out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        procs.append(GpuProcess(pid=_parse_int(parts[0]), used_mb=_parse_int(parts[1]), name=parts[2]))
    return procs
