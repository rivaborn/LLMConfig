"""A cookbook apply must confirm an unload SETTLED before reloading the node.

Regression from 2026-08-05. A DF4_RAG apply unloaded a 122B off spark4 and
launched the pooling pair into the gap immediately; both failed, from the two
sides of one race:

    reranker: OSError [Errno 98] Address already in use
    embedder: "0.80/0.95 is already committed to qwen35-122b_8_3_26"

Both succeeded on retry with nothing else changed. GPU lanes never hit this —
`_wait_vram_free` already applies the rule. Sparks had no equivalent.
"""
import asyncio

import pytest

from llmconfig.cookbook import Cookbook


class FakeCfg:
    def __init__(self, uid):
        self.id = uid
        self.enabled = True


class FakeLoaded:
    def __init__(self, model, server="spark"):
        self.model = model
        self.server = server


class FakeStatus:
    def __init__(self, models):
        self.loaded_models = [FakeLoaded(m) for m in models]


class FakeUnit:
    """Reports `model` as resident for `linger` status() calls after unload."""

    kind = "spark"

    def __init__(self, resident, linger=0):
        self.cfg = FakeCfg("spark4")
        self._resident = list(resident)
        self._linger = linger
        self.status_calls = 0
        self.unloaded = []

    def canonical_model(self, m):
        return m

    async def status(self):
        self.status_calls += 1
        return FakeStatus(self._resident)

    async def unload(self, req):
        self.unloaded.append(req.model)
        # The defect being modelled: the call returns while the node still holds it.
        if self._linger <= 0:
            self._resident = [m for m in self._resident if m != req.model]
        else:
            async def release(model, after):
                for _ in range(after):
                    await asyncio.sleep(0)
                self._resident = [m for m in self._resident if m != model]
            asyncio.get_running_loop().create_task(release(req.model, self._linger))
        return None


class FakeJobs:
    def __init__(self):
        self.lines = []

    def log(self, job, msg):
        self.lines.append(msg)


class FakeSettings:
    evict_timeout_s = 0.5
    poll_interval_s = 0.01


def _book(jobs):
    cb = Cookbook.__new__(Cookbook)
    cb.s = FakeSettings()
    cb.jobs = jobs
    cb.orch = None
    cb.leases = None
    cb.path = None
    return cb


async def test_returns_once_the_model_is_gone():
    jobs = FakeJobs()
    unit = FakeUnit(["qwen35-122b_8_3_26"])
    unit._resident = []                      # already released
    await _book(jobs)._wait_unloaded(None, unit, ["qwen35-122b_8_3_26"])
    assert unit.status_calls == 1, "no polling needed when it is already gone"


async def test_waits_while_the_node_still_holds_it():
    """The actual bug: unload() returned, residency lingered."""
    jobs = FakeJobs()
    unit = FakeUnit(["qwen35-122b_8_3_26"], linger=3)

    class Req:
        model = "qwen35-122b_8_3_26"

    await unit.unload(Req())
    await _book(jobs)._wait_unloaded(None, unit, ["qwen35-122b_8_3_26"])
    assert unit._resident == []
    assert unit.status_calls > 1, "must have polled rather than trusting the return"


async def test_timeout_is_non_fatal_and_says_so():
    """Losing the wait must not be worse than never having had it."""
    jobs = FakeJobs()
    unit = FakeUnit(["stuck-model"])          # never releases
    await _book(jobs)._wait_unloaded(None, unit, ["stuck-model"])
    assert any("still resident after" in l for l in jobs.lines)
    assert any("loading anyway" in l for l in jobs.lines)


async def test_probe_failure_does_not_break_the_apply():
    jobs = FakeJobs()

    class Boom(FakeUnit):
        async def status(self):
            raise RuntimeError("node unreachable")

    await _book(jobs)._wait_unloaded(None, Boom(["x"]), ["x"])
    assert any("residency probe failed" in l for l in jobs.lines)


async def test_only_the_named_models_are_waited_on():
    """A co-resident that is STAYING must not hold the wait open forever."""
    jobs = FakeJobs()
    unit = FakeUnit(["going", "staying"])
    unit._resident = ["staying"]              # 'going' already released
    await _book(jobs)._wait_unloaded(None, unit, ["going"])
    assert not any("still resident" in l for l in jobs.lines)
