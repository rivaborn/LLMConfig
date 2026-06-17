"""In-memory async job tracking for long operations (vLLM loads, Ollama pulls).

Endpoints kick off a job and return its id immediately; clients poll
GET /api/jobs/{id}. Jobs live for the process lifetime (bounded history).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Awaitable, Callable, Optional

from .schemas import Job

JobBody = Callable[[Job], Awaitable[Optional[dict]]]
_MAX_HISTORY = 50


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, kind: str) -> Job:
        jid = uuid.uuid4().hex[:12]
        self._jobs[jid] = Job(id=jid, kind=kind, state="pending", created_at=time.time())
        self._prune()
        return self._jobs[jid]

    def get(self, jid: str) -> Optional[Job]:
        return self._jobs.get(jid)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def log(self, job: Job, line: str) -> None:
        line = line.strip()
        if not line:
            return
        job.log.append(line)
        job.message = line
        if len(job.log) > 200:
            job.log = job.log[-200:]

    def start(self, job: Job, body: JobBody) -> Job:
        """Run `body` as a background task that drives this job to a terminal state."""
        job.state = "running"
        self._tasks[job.id] = asyncio.create_task(self._run(job, body))
        return job

    async def _run(self, job: Job, body: JobBody) -> None:
        try:
            result = await body(job)
            job.result = result or {}
            job.state = "succeeded"
        except asyncio.CancelledError:
            job.error = "cancelled"
            job.state = "failed"
            raise
        except Exception as e:  # surface, don't crash the server
            job.error = f"{type(e).__name__}: {e}"
            job.state = "failed"
            if not job.message:
                job.message = job.error
        finally:
            job.finished_at = time.time()
            self._tasks.pop(job.id, None)

    def _prune(self) -> None:
        if len(self._jobs) <= _MAX_HISTORY:
            return
        terminal = [j for j in self.list() if j.state in ("succeeded", "failed")]
        for j in terminal[_MAX_HISTORY:]:
            self._jobs.pop(j.id, None)
