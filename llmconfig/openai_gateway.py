"""OpenAI-compatible `/v1` gateway — auto-loads the requested model, then proxies.

opencode's `/model` picker has no selection-time hook, so the model switch must
happen on the **inference path**: the first request for a model triggers the load.
This router resolves the requested `model` to a lane backend, ensures it's loaded via
the existing per-lane arbitration (`orch.load` → a Job), **streams the load progress**
to the client on a cold load, then reverse-proxies the request to the real backend.

No new arbitration: it reuses the lane lock / eviction / WSL-keepalive that
`Lane.load` already performs. It just moves the client-side `vllm-swap` poll loop
(resolve → load → poll → forward) server-side and adds SSE progress.

Lane selection: the `X-LLM-Lane` header (default `primary`). opencode's `companion`
provider sets `X-LLM-Lane: companion` so its models land on the RTX 3070 Ti.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .jobs import JobManager
from .lane import Lane
from .orchestrator import Orchestrator, Unit
from .placement import classify_workload, wants_auto, workload_priority
from .schemas import LeaseClaimRequest, LoadRequest
from .spark_unit import SparkUnit

if TYPE_CHECKING:
    from .leases import LeaseManager
    from .placement import Placer


def _spark_shaped(lane) -> bool:
    """True for a single Spark node AND for a multi-node `SparkGroup`.

    Everything this gateway needs from "a Spark" — a curated catalog to resolve
    against, per-model port routing, `cfg.api_base`, a minutes-long load timeout,
    and NO ollama/vllm halves — is equally true of a group, which serves its
    tensor-parallel model from the head's port. Using `isinstance(SparkUnit)`
    alone sent groups down the LANE branch, where `cfg.vllm_relay_url` does not
    exist — an AttributeError on /v1/models the moment a group unit exists."""
    return isinstance(lane, SparkUnit) or getattr(lane, "kind", "") == "spark_group"


class OpenAIGateway:
    """Holds the long-lived forwarding client + the resolve/load/forward logic."""

    def __init__(self, orch: Orchestrator, jobs: JobManager, settings: Settings,
                 leases: "LeaseManager | None" = None,
                 placer: "Placer | None" = None,
                 stats=None):
        self.orch = orch
        self.jobs = jobs
        self.s = settings
        self.leases = leases
        # None (or AUTO_PLACE_ENABLED=false) disables auto-placement: a request
        # without X-LLM-Lane then falls back to "primary" exactly as before.
        self.placer = placer
        self.stats = stats   # UsageStats — one note_request per routed request
        self._http: httpx.AsyncClient | None = None

    # ---- forwarding client (no read timeout: chat generations can run long) ----
    def client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def lane(self, lane_id: str) -> "Unit":
        try:
            return self.orch.lane(lane_id)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def resolve(self, lane: "Unit", model: str) -> Optional[tuple[str, str, str]]:
        """Map an OpenAI `model` id → (server, load_arg, backend_base_url).

        - on a Spark unit: a curated `served_name` → ("spark", alias, node slot-0 base)
          The URL here is only a DEFAULT: a Spark can serve several models at once,
          each on its own port, and which port a model occupies is not known until
          status() has probed. `_route()` corrects it once residency is known.
        - a vLLM `served_name` (uses `-`) → ("vllm", alias, lane relay url)
        - else an Ollama tag (has `:`)     → ("ollama", tag, lane ollama url)
        - else None (→ 404). The id formats don't collide.
        """
        if not model:
            return None
        # A Spark's catalog is the only thing to match; the port comes later.
        if _spark_shaped(lane):
            fallback: Optional[str] = None
            for e in lane.registry.entries():
                if (e.served_name or e.alias) == model:
                    if e.status != "blocked":
                        return ("spark", e.alias, lane.cfg.api_base)
                    fallback = fallback or e.alias
            if fallback is not None:
                return ("spark", fallback, lane.cfg.api_base)
            return None
        # vLLM: match the served_name; prefer a non-blocked alias if several ever share one.
        # Skipped entirely on an Ollama-only lane — resolving there would send the
        # request down a load path that cannot succeed.
        # A SlotLane serves N models on N relays, so the relay is per-ALIAS there
        # (`relay_url_for`); a classic Lane keeps its single configured relay.
        def _relay(alias: str) -> str:
            fn = getattr(lane, "relay_url_for", None)
            return fn(alias) if fn is not None else lane.cfg.vllm_relay_url

        match: Optional[str] = None
        for e in (lane.registry.entries() if lane.cfg.vllm_enabled else []):
            if (e.served_name or e.alias) == model:
                if e.status != "blocked":
                    return ("vllm", e.alias, _relay(e.alias))
                match = match or e.alias
        if match is not None:
            return ("vllm", match, _relay(match))
        # Ollama: a tag present in the lane's catalog
        if ":" in model:
            if self.placer is not None:
                names = await self.placer._ollama_tags(lane)   # TTL + negative cache
            else:
                try:
                    names = {m.name for m in await lane.ollama.list_models()}
                except Exception:
                    names = set()
            if model in names:
                return ("ollama", model, lane.cfg.ollama_url)
        return None

    @staticmethod
    def _resident(status, model: str):
        """The resident `LoadedModel` for `model` on this unit, or None.

        Reads the additive `loaded_models` list rather than the scalar `loaded`,
        which only ever names the unit's primary occupant — the reason a request
        for a co-resident model used to miss the fast path and tear down its
        neighbour to "load" something that was already up.
        """
        for m in (status.loaded_models or []):
            if m.model == model:
                return m
        return None

    def _route(self, lane: "Unit", server: str, model: str, status, default: str) -> str:
        """Backend URL for THIS model.

        Only Sparks can disagree with the unit-wide default: several models, one
        port each. Falling back to `default` (slot 0) when the model is not
        resident is what lets a cold load still reach a backend.
        """
        if server != "spark":
            return default
        m = self._resident(status, model)
        if m is not None and m.port:
            return lane.cfg.api_base_for(m.port)
        return default

    # ---- lease admission (see leases.py) ----
    def _lease_gate(self, unit_id: str, lease_id: str, model: str = ""):
        """Decide whether this request may use the unit. Returns None to allow, or
        `(code, message, lease)` to refuse.

        Fully synchronous on purpose: it runs before any await in the request path,
        so a lease cannot be claimed or revoked underneath the decision.

        Un-leased traffic keeps working (opencode sends no lease header) EXCEPT
        against a non-preemptible claim — that is the one behavioural change, and
        `LEASE_BLOCK_UNLEASED=false` disables it.
        """
        if self.leases is None or not self.s.lease_enabled:
            return None

        if not lease_id:
            blocker = self.leases.blocks_unleased(unit_id, model or None)
            if blocker is None:
                return None
            what = f"model '{blocker.model}' on {unit_id}" if blocker.model else f"unit '{unit_id}'"
            return ("lease_required",
                    f"{what} is held by '{blocker.holder}' with a "
                    f"non-preemptible lease — claim one (POST /api/leases) or wait",
                    blocker)

        lease = self.leases.get(lease_id)
        if lease is None:
            reason = ("server_restarted" if self.leases.looks_like_pre_restart(lease_id)
                      else "unknown")
            return ("lease_revoked",
                    f"lease {lease_id} is not known to this server ({reason}) — re-claim and retry",
                    None)
        # Terminal states FIRST: a lease revoked by an eviction and then re-placed
        # onto another unit must answer `lease_revoked` (who took it and why),
        # not a confusing `lease_wrong_unit` about a claim that no longer exists.
        if lease.state == "revoked":
            who = f" by '{lease.revoked_by}'" if lease.revoked_by else ""
            why = f" ({lease.revoked_reason})" if lease.revoked_reason else ""
            return ("lease_revoked",
                    f"lease {lease.id} was revoked{who}{why} — re-claim before retrying", lease)
        if lease.state == "released":
            return ("lease_released",
                    f"lease {lease.id} was released — claim a new one", lease)
        if lease.unit != unit_id:
            return ("lease_wrong_unit",
                    f"lease {lease_id} is for unit '{lease.unit}', not '{unit_id}'", lease)

        now_mono = time.monotonic()
        if lease.is_live(now_mono):
            self.leases.note_request(lease.id)
            return None
        # Past the deadline but inside the grace window: still serve this holder (its
        # renew timer may just have slipped behind a long generation). Note the
        # asymmetry — it no longer blocks anyone else.
        if lease.state in ("active", "expired") and lease.in_grace(
                now_mono, self.s.lease_expiry_grace_s):
            self.leases.note_request(lease.id)
            return None
        return ("lease_expired",
                f"lease {lease.id} expired — renew sooner or claim a new one", lease)

    def _lease_reject(self, code: str, message: str, lease, *, is_chat: bool, stream: bool):
        """Refuse a request that failed the lease gate.

        A rejected stream gets a real HTTP 409 by default: the gate decides before a
        single byte is written, so the status line is still available. (The existing
        `❌ load failed` case is different — bytes had already gone out.) Set
        LEASE_STREAM_REJECT_MODE=sse for clients that can't surface a non-200 here.
        """
        payload = {"error": {"message": message, "type": "invalid_request_error", "code": code}}
        if lease is not None:
            # Additive sibling — strict OpenAI clients read error.message and ignore this.
            payload["error"]["lease"] = {
                "id": lease.id, "state": lease.state, "unit": lease.unit,
                "holder": lease.holder, "revoked_by": lease.revoked_by,
                "revoked_reason": lease.revoked_reason, "revoked_at": lease.revoked_at,
            }
        if stream and self.s.lease_stream_reject_mode == "sse":
            cid = ("chatcmpl-" if is_chat else "cmpl-") + uuid.uuid4().hex[:12]
            created = int(time.time())

            async def _gen():
                yield self._progress_chunk(cid, created, "", f"❌ {message}\n", is_chat)
                yield self._final_chunk(cid, created, "", is_chat)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return JSONResponse(status_code=409, content=payload)

    async def _choose(self, request: Request, model: str, *, is_chat: bool,
                      stream: bool, body: dict | None = None):
        """Pick the unit for this request. Returns
        (lane, victims, group_victims, lease_id, priority) or an error Response.
        The ONE place auto-placement is wired, shared by the chat and pooling
        handlers so the two cannot drift.

        `priority` is the request's classified placement priority for a "place"
        decision, else None: an explicit header, a lease pin, and a sole-
        candidate PIN all keep the unit's own load semantics (a lane's
        last-writer-wins eviction included), so their loads carry no priority
        and skip the unit-side active-preemption re-validation.
        `group_victims` is executed by the caller via
        `placer.evict_group_victims` BEFORE the load job (a group teardown
        needs the target member's lock — running it inside the load would
        deadlock).

        Explicit `X-LLM-Lane: <unit>` pins, exactly as before. Absent/empty/`auto`
        auto-places — unless a valid `X-LLM-Lease` names a unit, in which case the
        lease IS the placement (a lease is a claim on that unit; scattering its
        holder elsewhere would defeat it). An invalid lease id does NOT disable
        auto: place normally, then the lease gate still reports
        `server_restarted`/`lease_revoked` with the id the client sent.

        Placement is advisory, so a lease landing on the chosen unit between the
        sweep and the gate yields a refusal — retried ONCE with that unit
        excluded before giving up (more would oscillate).
        """
        header = request.headers.get("x-llm-lane")
        lease_id = (request.headers.get("x-llm-lease") or "").strip()
        auto = (self.placer is not None and self.s.auto_place_enabled
                and wants_auto(header))
        # Workload tiering: interactive → the 3090 (speed tier), batch → the
        # Sparks (capacity tier). Classified from the body, X-LLM-Workload
        # header wins; only a preference inside rank(), never a gate.
        workload = classify_workload(body or {}, request.headers.get("x-llm-workload"),
                                     self.s) if auto else None
        if not auto:
            lane = self.lane(header or "primary")
            reject = self._lease_gate(lane.cfg.id, lease_id, model)
            if reject is not None:
                return self._lease_reject(*reject, is_chat=is_chat, stream=stream)
            return lane, [], [], lease_id, None

        if lease_id and self.leases is not None:
            lease = self.leases.get(lease_id)
            # Pin while ACTIVE or inside the expiry grace window — _lease_gate
            # still serves an in-grace holder on its own unit, so auto-placing
            # it elsewhere would hand it a lease_wrong_unit 409 (the exact
            # renew-slipped-behind-a-long-generation case grace exists for).
            in_grace = (lease is not None and lease.state in ("active", "expired")
                        and lease.in_grace(time.monotonic(), self.s.lease_expiry_grace_s))
            if lease is not None and (lease.state == "active" or in_grace):
                lane = self.lane(lease.unit)
                reject = self._lease_gate(lane.cfg.id, lease_id, model)
                if reject is not None:
                    return self._lease_reject(*reject, is_chat=is_chat, stream=stream)
                return lane, [], [], lease_id, None

        # Session affinity: go back to the unit this holder already holds for this
        # model, so consecutive requests stay on one warm prefix cache.
        holder = (request.headers.get("x-llm-hold") or "").strip()[:64]
        if holder and self.s.auto_hold_enabled:
            uid = self._hold_unit(holder, model)
            if uid is not None:
                lane = self.lane(uid)
                reject = self._lease_gate(lane.cfg.id, lease_id, model)
                if reject is None:
                    return lane, [], [], lease_id, None

        exclude: frozenset[str] = frozenset()
        for attempt in range(2):
            decision = await self.placer.place(model, exclude=exclude,
                                               workload=workload)
            if decision.outcome == "not_found":
                return JSONResponse(
                    status_code=404,
                    content={"error": {"message": f"model '{model}' not found on any unit",
                                        "type": "invalid_request_error",
                                        "code": "model_not_found"}},
                )
            if decision.outcome == "no_capacity":
                why = "; ".join(f"{u}: {r}" for u, r in sorted(decision.reasons.items()))
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": f"no unit can take '{model}' right now — {why}",
                                        "type": "server_error", "code": "no_capacity",
                                        "reasons": decision.reasons}},
                )
            lane = self.lane(decision.unit_id)
            reject = self._lease_gate(lane.cfg.id, lease_id, model)
            if reject is None:
                prio = (workload_priority(workload, self.s)
                        if decision.outcome == "place" else None)
                return (lane, list(decision.victims),
                        list(decision.group_victims), lease_id, prio)
            if attempt == 0:
                exclude = frozenset({decision.unit_id})
                continue
            return self._lease_reject(*reject, is_chat=is_chat, stream=stream)

    def _hold_unit(self, holder: str, model: str) -> Optional[str]:
        """The unit where `holder` already holds `model`, if any (sync, dict scan).

        Session affinity. Without it, "idle beats active" ping-pongs a client
        between two units serving the same model: your own request makes unit A
        active, so your NEXT request prefers idle unit B, and so on — observed
        live, one opencode turn leaving leases on two Sparks. A holder that is
        already holding this model somewhere has a warm prefix cache there and
        should go back to it.
        """
        if self.leases is None or not holder:
            return None
        for lease in self.leases.list(active_only=True):
            if lease.holder == holder and lease.model and lease.unit in self.orch.units:
                if self.leases._canon(lease.unit, model) == lease.model:
                    return lease.unit
        return None

    def _requester(self, request: Request) -> str:
        """Who to name in `revoked_by` when this request's load displaces someone:
        the X-LLM-Hold holder, else X-LLM-App, else the holder of a known
        X-LLM-Lease, else 'placement'. Sync dict reads only."""
        holder = (request.headers.get("x-llm-hold") or "").strip()[:64]
        if holder:
            return holder
        app = (request.headers.get("x-llm-app") or "").strip()[:64]
        if app:
            return app
        lease_id = (request.headers.get("x-llm-lease") or "").strip()
        if lease_id and self.leases is not None:
            lease = self.leases.get(lease_id)
            if lease is not None and lease.holder:
                return lease.holder
        return "placement"

    def _client_identity(self, request: Request) -> tuple[str, str, str]:
        """(app, function, ip) — who is behind this request, for attribution.

        `app` resolves X-LLM-App -> the X-LLM-Hold name -> the holder of a
        valid X-LLM-Lease -> "" (the strict gate and the stats layer render
        that as "unknown"). `ip` is captured SERVER-SIDE from the connection —
        uvicorn serves directly, so the peer address is the real client and
        even a silent browser is attributed to a machine. Built 2026-08-08,
        after naming the launcher of a mystery qwen35-122b took a netstat hunt
        across three machines. Sync dict/header reads only."""
        app = (request.headers.get("x-llm-app") or "").strip()[:64]
        fn = (request.headers.get("x-llm-function") or "").strip()[:120]
        ip = (request.client.host if request.client else "") or ""
        if not app:
            app = (request.headers.get("x-llm-hold") or "").strip()[:64]
        if not app and self.leases is not None:
            lid = (request.headers.get("x-llm-lease") or "").strip()
            if lid:
                lease = self.leases.get(lid)
                if lease is not None and lease.holder:
                    app = lease.holder
        return app, fn, ip

    def _identity_reject(self, app: str):
        """400 for an anonymous request when CLIENT_ID_REQUIRED is on, else None.

        Actionable by design: the message names the header to send. Never gates
        read-only endpoints — only the paths that consume an LLM unit."""
        if not self.s.client_id_required or app:
            return None
        return JSONResponse(status_code=400, content={"error": {
            "message": "this gateway requires clients to identify themselves — "
                       "send X-LLM-App: <app-name> (and optionally "
                       "X-LLM-Function: <purpose>) on every request",
            "type": "invalid_request_error", "code": "client_id_required"}})

    def _auto_hold(self, lane: "Unit", model: str, request: Request,
                   priority: int | None = None) -> None:
        """Honour `X-LLM-Hold: <holder>` — claim/renew this holder's lease on the
        model it is actually using.

        A static client config cannot carry a lease id (one does not exist until
        claimed), so this is how a config-only client gets a lease at all. The
        claim is **preemptible** and carries the request's classified placement
        priority: it stops the idle reaper, and against auto-placement it yields
        once idle (or, while active, to strictly higher-priority traffic — see
        `_victim_eligible`), with the revocation telling the holder why. That is
        what "don't displace my session" now means; refusing other traffic
        outright stays a manual, non-preemptible /api/leases claim.

        Best-effort and never raises into the request: a hold is a convenience,
        not the point of the call. Never preempts AT CLAIM TIME — if someone
        else already holds this model, we leave it alone; their claim is at
        least as strong as ours.
        """
        holder = (request.headers.get("x-llm-hold") or "").strip()[:64]
        if not holder or self.leases is None:
            return
        if not (self.s.lease_enabled and self.s.auto_hold_enabled):
            return
        try:
            # Fold to the canonical alias before the never-steal check: leases
            # store canonical names, and the raw served-name miss used to be
            # masked only by claim()'s equal-priority refusal — with classified
            # priorities on the claim it would actually steal.
            existing = self.leases.active_for(
                lane.cfg.id, self.leases._canon(lane.cfg.id, model))
            if existing is not None and existing.holder != holder:
                return
            self.leases.claim(LeaseClaimRequest(
                unit=lane.cfg.id, holder=holder, model=model,
                preemptible=True, ttl_s=self.s.auto_hold_ttl_s,
                priority=(priority if priority is not None
                          else self.s.placement_priority_neutral),
                note="auto-hold (X-LLM-Hold)",
            ))
        except Exception:  # noqa: BLE001 — a failed hold must never fail the request
            pass

    @staticmethod
    def _unit_headers(lane: "Unit") -> dict:
        """`X-LLM-Unit` on every success response — with auto-placement the client
        no longer knows where its request ran unless we say so."""
        return {"x-llm-unit": lane.cfg.id}

    def _ensure_load_job(self, lane: Lane, status, target_kind: str, server: str,
                         load_arg: str, stream: bool, evict: list[str] | None = None,
                         priority: int | None = None, requested_by: str = ""):
        """Return (job_or_None, short_circuit). Coalesces onto an identical in-flight
        load; for a *different* in-flight load, queues ours (stream) or signals a
        short-circuit (non-stream, so title-gen doesn't block for minutes).

        The short-circuit exists because a GPU lane holds ONE model: a different
        load in flight means the model being asked for is about to be evicted, so
        waiting is pointless. That is not true of a Spark, where the in-flight load
        is for another slot entirely and has no bearing on this request — hence the
        `isinstance` guard below.
        """
        if status.swap_in_progress and status.active_job_id:
            active = self.jobs.get(status.active_job_id)
            if active and active.kind == target_kind and active.state in ("pending", "running"):
                return active, False  # identical target already loading → attach
            # Deliberately `isinstance`, NOT `_spark_shaped`: a SparkGroup gets the
            # LANE treatment here. Co-residency is what makes another Spark load
            # irrelevant, and a group has none — its load claims every member for
            # many minutes, so a non-stream client is better off failing fast than
            # queueing behind it.
            if not stream and not isinstance(lane, SparkUnit):
                return None, True     # different model loading + non-stream → bail fast
            # stream, or a Spark (co-residency makes the other load irrelevant):
            # queue ours behind the unit lock (shows "waiting…")
        job = self.orch.load(LoadRequest(server=server, model=load_arg, lane=lane.cfg.id,
                                         evict=list(evict or []),
                                         priority=priority, requested_by=requested_by))
        # A load just committed: drop the placer's cached sweep, so a burst's
        # next auto request re-sweeps and sees this in-flight job instead of
        # double-placing onto the same unit off the stale snapshot.
        if self.placer is not None:
            self.placer.invalidate()
        return job, False

    def _fwd_headers(self, headers) -> dict:
        """Forward only what the upstream needs. Pass an Authorization through so a
        keyed backend still works; drop hop-by-hop / length headers (httpx resets)."""
        out = {"content-type": "application/json"}
        auth = headers.get("authorization")
        if auth:
            out["authorization"] = auth
        return out

    @staticmethod
    def _backend_url(backend: str, sub_path: str) -> str:
        return backend.rstrip("/") + "/v1" + sub_path

    # ---- non-streaming forward ----
    async def forward(self, backend: str, sub_path: str, body: dict, headers,
                      extra_headers: dict | None = None) -> Response:
        url = self._backend_url(backend, sub_path)
        try:
            resp = await self.client().post(url, json=body, headers=self._fwd_headers(headers))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}: {e}")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers=extra_headers or None,
        )

    # ---- streaming forward (raw passthrough, incl. upstream's `data: [DONE]`) ----
    async def _forward_stream(self, backend: str, sub_path: str, body: dict, headers):
        url = self._backend_url(backend, sub_path)
        async with self.client().stream(
            "POST", url, json=body, headers=self._fwd_headers(headers)
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                detail = raw.decode("utf-8", "ignore")[:500] or f"HTTP {resp.status_code}"
                raise RuntimeError(f"upstream {resp.status_code}: {detail}")
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------ #
    # SSE chunk builders (OpenAI chat.completion.chunk / text_completion)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sse(payload: dict) -> bytes:
        return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")

    def _progress_chunk(self, cid: str, created: int, model: str, text: str, is_chat: bool) -> bytes:
        if is_chat:
            choice = {"index": 0, "delta": {"content": text}, "finish_reason": None}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": text, "finish_reason": None}
            obj = "text_completion"
        return self._sse({"id": cid, "object": obj, "created": created, "model": model,
                          "choices": [choice]})

    def _final_chunk(self, cid: str, created: int, model: str, is_chat: bool) -> bytes:
        if is_chat:
            choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": "", "finish_reason": "stop"}
            obj = "text_completion"
        return self._sse({"id": cid, "object": obj, "created": created, "model": model,
                          "choices": [choice]})

    def _minimal_completion(self, model: str, is_chat: bool) -> dict:
        """A valid, empty 200 so a non-stream caller (e.g. opencode title-gen) returns
        immediately instead of blocking for minutes on a cold load it didn't ask for."""
        created = int(time.time())
        cid = ("chatcmpl-" if is_chat else "cmpl-") + uuid.uuid4().hex[:12]
        if is_chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": "", "finish_reason": "stop"}
            obj = "text_completion"
        return {"id": cid, "object": obj, "created": created, "model": model, "choices": [choice],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    # ------------------------------------------------------------------ #
    # Job polling
    # ------------------------------------------------------------------ #
    async def _wait_job(self, job_id: str, timeout: float) -> tuple[bool, str]:
        """Block (non-stream path) until the load Job is terminal. → (ok, error)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cur = self.jobs.get(job_id)
            if cur is None:
                return False, "load job vanished"
            if cur.state == "succeeded":
                return True, ""
            if cur.state == "failed":
                return False, cur.error or cur.message or "load failed"
            await asyncio.sleep(1.0)
        cur = self.jobs.get(job_id)   # one last read — the job may have flipped
        if cur is not None and cur.state == "succeeded":   # during the final sleep
            return True, ""
        return False, f"load did not finish within {int(timeout)}s"

    def _reroute(self, lane: "Unit", server: str, model: str, default: str):
        """Late-bound backend resolution for a cold Spark load.

        Which slot a cold load lands on (lowest FREE port wins) is unknowable
        until the job finishes, so the forward target cannot be computed up
        front — the pre-load answer is always slot 0, which on a multi-model
        node is usually some OTHER model's port. Callers await this AFTER the
        load succeeds, when residency finally says where the model went.
        """
        async def _resolve() -> str:
            return self._route(lane, server, model, await lane.status(), default)
        return _resolve

    async def _stream_load_then_forward(self, job_id: Optional[str], backend: str, sub_path: str,
                                        body: dict, model: str, headers, is_chat: bool,
                                        lane: Optional[Lane] = None, reroute=None):
        """Wrapper that marks the lane active when the stream finishes (however it
        ends), so a generation longer than the idle timeout isn't reaped mid-answer
        even when the Monitor's util signal is unavailable."""
        try:
            async for chunk in self._stream_load_then_forward_inner(
                job_id, backend, sub_path, body, model, headers, is_chat, reroute
            ):
                yield chunk
        finally:
            if lane is not None:
                lane.touch(model=model)

    async def _stream_load_then_forward_inner(self, job_id: Optional[str], backend: str,
                                              sub_path: str, body: dict, model: str,
                                              headers, is_chat: bool, reroute=None):
        created = int(time.time())
        cid = ("chatcmpl-" if is_chat else "cmpl-") + uuid.uuid4().hex[:12]
        emitted = 0
        if job_id is not None:
            while True:
                cur = self.jobs.get(job_id)
                if cur is None:
                    yield self._progress_chunk(cid, created, model, "\nload job vanished\n", is_chat)
                    yield self._final_chunk(cid, created, model, is_chat)
                    yield b"data: [DONE]\n\n"
                    return
                while emitted < len(cur.log):
                    line = cur.log[emitted]
                    emitted += 1
                    yield self._progress_chunk(cid, created, model, f"⏳ {line}\n", is_chat)
                if cur.state == "succeeded":
                    break
                if cur.state == "failed":
                    err = cur.error or cur.message or "load failed"
                    yield self._progress_chunk(cid, created, model, f"❌ load failed: {err}\n", is_chat)
                    yield self._final_chunk(cid, created, model, is_chat)
                    yield b"data: [DONE]\n\n"
                    return
                await asyncio.sleep(1.5)
            # The load is done — only now does residency say which slot it took
            # (see `_reroute`); the `backend` computed before the load is slot 0.
            if reroute is not None:
                backend = await reroute()
        # Loaded — relay the upstream completion verbatim (it emits its own [DONE]).
        try:
            async for chunk in self._forward_stream(backend, sub_path, body, headers):
                yield chunk
        except Exception as e:  # noqa: BLE001 — surface upstream failure into the stream
            yield self._progress_chunk(cid, created, model, f"\n❌ upstream error: {e}\n", is_chat)
            yield self._final_chunk(cid, created, model, is_chat)
            yield b"data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # Request handler (shared by chat/completions and completions)
    # ------------------------------------------------------------------ #
    async def handle_completion(self, request: Request, sub_path: str, is_chat: bool):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        model = body.get("model") or ""
        stream = bool(body.get("stream"))

        # Identity FIRST: the strict gate must refuse before placement runs any
        # side effects, and the capture is needed for attribution either way.
        app, fn, ip = self._client_identity(request)
        reject = self._identity_reject(app)
        if reject is not None:
            return reject

        # Admission + placement BEFORE resolve() and before the first lane.touch():
        # the decision shouldn't depend on model resolution, and a refused request
        # must not extend anyone's idle window (it never reached a backend).
        chosen = await self._choose(request, model, is_chat=is_chat, stream=stream,
                                    body=body)
        if isinstance(chosen, Response):
            return chosen
        lane, victims, group_victims, _lease_id, priority = chosen
        requested_by = self._requester(request)

        resolved = await self.resolve(lane, model)
        if resolved is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"model '{model}' not found on lane '{lane.cfg.id}'",
                                    "type": "invalid_request_error", "code": "model_not_found"}},
            )
        server, load_arg, backend = resolved
        lane.touch(model=model)  # inference traffic — reset this model's idle-unload window
        # The hold claims at the request's CLASSIFIED priority (independent of
        # which placement tier fired) so later contention reflects the traffic.
        wl = classify_workload(body or {}, request.headers.get("x-llm-workload"),
                               self.s)
        if self.stats is not None:   # once per request, not per touch
            self.stats.note_request(lane.cfg.id, model, wl.cls if wl else "",
                                    client=app, ip=ip, fn=fn)
        self._auto_hold(lane, model, request,
                        priority=workload_priority(wl, self.s))

        # Group victims first — the teardown holds the target member's lock, so
        # it must run BEFORE the load job, from up here (see _choose). A
        # conflict means the world moved: one re-place (non-stream), else 503.
        if group_victims:
            try:
                await self.placer.evict_group_victims(
                    group_victims, priority=priority, requested_by=requested_by)
            except RuntimeError as e:
                if not stream and any(m in str(e) for m in self._CONFLICT_MARKERS):
                    retry = await self._retry_elsewhere(request, model, str(e), lane,
                                                        sub_path, body, is_chat=is_chat)
                    if retry is not None:
                        return retry
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": f"could not free the nodes for "
                                                  f"'{model}': {e}",
                                        "type": "server_error", "code": "no_capacity"}},
                )

        # Fast path: this model is ALREADY resident → forward, no load. Membership
        # over loaded_models, not equality against the unit's primary occupant: on a
        # multi-model Spark the latter made a request for co-resident model B miss
        # the fast path and tear down model A to reload B.
        status = await lane.status()
        backend = self._route(lane, server, model, status, backend)
        if any(m.server == server and m.model == model for m in status.loaded_models):
            if stream:
                return StreamingResponse(
                    self._stream_load_then_forward(None, backend, sub_path, body, model,
                                                   request.headers, is_chat, lane=lane),
                    media_type="text/event-stream",
                    headers=self._unit_headers(lane),
                )
            resp = await self.forward(backend, sub_path, body, request.headers,
                                      extra_headers=self._unit_headers(lane))
            lane.touch(model=model)  # generation finished — a long answer shouldn't count as idle time
            return resp

        # Need to load (cold / wrong model). Coalesce onto an identical in-flight load.
        target_kind = f"load:{lane.cfg.id}:{server}:{load_arg}"
        job, short_circuit = self._ensure_load_job(lane, status, target_kind, server, load_arg,
                                                   stream, evict=victims,
                                                   priority=priority,
                                                   requested_by=requested_by)

        if stream:
            reroute = (self._reroute(lane, server, model, backend)
                       if _spark_shaped(lane) else None)
            return StreamingResponse(
                self._stream_load_then_forward(job.id if job else None, backend, sub_path, body,
                                               model, request.headers, is_chat, lane=lane,
                                               reroute=reroute),
                media_type="text/event-stream",
                headers=self._unit_headers(lane),
            )

        # Non-stream: a different model is mid-load → return an empty 200 (don't hang).
        if short_circuit or job is None:
            return JSONResponse(content=self._minimal_completion(model, is_chat))

        if server == "spark":
            entry = lane.registry.get(load_arg)
            # Spark cold starts pull large weights over the network — give them the
            # per-recipe budget plus slack rather than the generic 600 s.
            timeout = float(entry.load_timeout_s if entry else lane.cfg.load_timeout_s) + 60.0
        elif server == "vllm":
            entry = lane.registry.get(load_arg)
            base = float(entry.load_timeout_s if entry else self.s.default_vllm_load_timeout_s)
            timeout = base + self.s.vllm_ready_grace_s + 30.0
        else:
            timeout = 600.0
        ok, err = await self._wait_job(job.id, timeout)
        if not ok:
            retry = await self._retry_elsewhere(request, model, err, lane,
                                                sub_path, body, is_chat=is_chat)
            if retry is not None:
                return retry
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"failed to load '{model}': {err}",
                                    "type": "server_error", "code": "model_load_failed"}},
            )
        if _spark_shaped(lane):
            # The cold load may have landed on any free slot; the `backend`
            # computed before it is slot 0. Re-route from fresh residency, or the
            # forward goes to whichever model already lives there.
            backend = self._route(lane, server, model, await lane.status(), backend)
        resp = await self.forward(backend, sub_path, body, request.headers,
                                  extra_headers=self._unit_headers(lane))
        lane.touch(model=model)  # generation finished — a long answer shouldn't count as idle time
        return resp

    # Error substrings that mean "the WORLD moved, not the model is broken":
    # a placement victim failed re-validation, or admission lost a budget race.
    _CONFLICT_MARKERS = ("placement_conflict:", "already committed to", "actually free")

    async def _retry_elsewhere(self, request: Request, model: str, err: str,
                               failed_lane: "Unit", sub_path: str, body: dict,
                               *, is_chat: bool):
        """One re-place on another unit after a conflict-class load failure.

        Only for AUTO requests (an explicit pin means the client chose), only for
        non-stream (a stream has already sent bytes), and only ONCE — the
        excluded-unit re-place either lands or the caller 503s with the original
        error. Returns a Response or None (= fall through to the 503)."""
        if self.placer is None or not self.s.auto_place_enabled:
            return None
        if not wants_auto(request.headers.get("x-llm-lane")):
            return None
        if not any(m in (err or "") for m in self._CONFLICT_MARKERS):
            return None
        workload = classify_workload(body or {},
                                     request.headers.get("x-llm-workload"), self.s)
        decision = await self.placer.place(
            model, exclude=frozenset({failed_lane.cfg.id}), workload=workload)
        if decision.outcome not in ("place", "pin"):
            return None
        lane = self.lane(decision.unit_id)
        # The gate must run here too: a sole-candidate PIN bypasses rank()'s
        # lease_refused predicate by design, so without this an excluded retry
        # could forward onto a non-preemptibly held unit with no 409.
        lease_id = (request.headers.get("x-llm-lease") or "").strip()
        if self._lease_gate(lane.cfg.id, lease_id, model) is not None:
            return None   # fall through to the original 503
        priority = (workload_priority(workload, self.s)
                    if decision.outcome == "place" else None)
        requested_by = self._requester(request)
        if decision.group_victims:
            # This IS the one re-place — a conflict here just falls through to
            # the caller's original 503, never a second retry.
            try:
                await self.placer.evict_group_victims(
                    decision.group_victims, priority=priority,
                    requested_by=requested_by)
            except RuntimeError:
                return None
        status = await lane.status()
        server, load_arg = decision.server, decision.load_arg
        backend = self._route(lane, server, model, status,
                              lane.cfg.api_base if _spark_shaped(lane)
                              else (lane.cfg.vllm_relay_url if server == "vllm"
                                    else lane.cfg.ollama_url))
        lane.touch(model=model)
        if not any(m.server == server and m.model == model for m in status.loaded_models):
            target_kind = f"load:{lane.cfg.id}:{server}:{load_arg}"
            job, short = self._ensure_load_job(lane, status, target_kind, server, load_arg,
                                               stream=False, evict=decision.victims,
                                               priority=priority,
                                               requested_by=requested_by)
            if short or job is None:
                return None
            entry = lane.registry.get(load_arg)
            timeout = float(getattr(entry, "load_timeout_s", 0) or lane.cfg.load_timeout_s
                            if _spark_shaped(lane) else 600.0) + 60.0
            ok, _err2 = await self._wait_job(job.id, timeout)
            if not ok:
                return None
            backend = self._route(lane, server, model, await lane.status(), backend)
        resp = await self.forward(backend, sub_path, body, request.headers,
                                  extra_headers=self._unit_headers(lane))
        lane.touch(model=model)
        return resp

    async def handle_pooling(self, request: Request, sub_path: str):
        """`/v1/embeddings` and `/v1/rerank` — pooling runners, not chat.

        Shares the whole admission path with `handle_completion` (lane header →
        lease gate → resolve → ensure-loaded → forward) but differs in two ways
        that matter:

        * **Never streams.** Pooling endpoints have no SSE form, so none of the
          progress-chunk machinery applies and a cold load simply blocks.
        * **Never fabricates a response.** The chat path answers a mid-load
          request with an empty `200` so opencode's title-gen doesn't hang for
          minutes — harmless there, but an empty embedding written into a vector
          store is silent data corruption, and an empty rerank silently reorders
          nothing. Every not-ready case here is an explicit 503 instead.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="request body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        model = body.get("model") or ""

        # Identity first — same reasoning as the chat path.
        app, fn, ip = self._client_identity(request)
        reject = self._identity_reject(app)
        if reject is not None:
            return reject

        # Same ordering as the chat path: admission + placement before resolve()
        # and before the first touch(). `stream` is always False — see docstring.
        chosen = await self._choose(request, model, is_chat=False, stream=False,
                                    body=body)
        if isinstance(chosen, Response):
            return chosen
        lane, victims, group_victims, _lease_id, priority = chosen
        requested_by = self._requester(request)

        resolved = await self.resolve(lane, model)
        if resolved is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"model '{model}' not found on lane '{lane.cfg.id}'",
                                    "type": "invalid_request_error", "code": "model_not_found"}},
            )
        server, load_arg, backend = resolved

        # Ollama's OpenAI surface has /v1/embeddings but no rerank endpoint. Say so
        # here rather than letting the backend return a confusing 404.
        if sub_path == "/rerank" and server == "ollama":
            return JSONResponse(
                status_code=501,
                content={"error": {"message": "rerank is not supported by the Ollama backend; "
                                              "use a vLLM or Spark pooling model",
                                    "type": "invalid_request_error", "code": "rerank_unsupported"}},
            )

        lane.touch(model=model)
        wl = classify_workload(body or {}, request.headers.get("x-llm-workload"),
                               self.s)
        if self.stats is not None:   # once per request, not per touch
            self.stats.note_request(lane.cfg.id, model, wl.cls if wl else "",
                                    client=app, ip=ip, fn=fn)
        self._auto_hold(lane, model, request,
                        priority=workload_priority(wl, self.s))

        # Group victims first — same protocol as the chat path (see _choose).
        if group_victims:
            try:
                await self.placer.evict_group_victims(
                    group_victims, priority=priority, requested_by=requested_by)
            except RuntimeError as e:
                if any(m in str(e) for m in self._CONFLICT_MARKERS):
                    retry = await self._retry_elsewhere(request, model, str(e), lane,
                                                        sub_path, body, is_chat=False)
                    if retry is not None:
                        return retry
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": f"could not free the nodes for "
                                                  f"'{model}': {e}",
                                        "type": "server_error", "code": "no_capacity"}},
                )

        status = await lane.status()
        backend = self._route(lane, server, model, status, backend)
        if any(m.server == server and m.model == model for m in status.loaded_models):
            resp = await self.forward(backend, sub_path, body, request.headers,
                                      extra_headers=self._unit_headers(lane))
            lane.touch(model=model)
            return resp

        target_kind = f"load:{lane.cfg.id}:{server}:{load_arg}"
        job, short_circuit = self._ensure_load_job(lane, status, target_kind, server, load_arg,
                                                   stream=False, evict=victims,
                                                   priority=priority,
                                                   requested_by=requested_by)
        if short_circuit or job is None:
            # A *different* model is mid-load. Never answer with an empty vector.
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"unit '{lane.cfg.id}' is loading another model; "
                                              f"retry once '{model}' is resident",
                                    "type": "server_error", "code": "unit_busy_loading"}},
            )

        if server == "spark":
            entry = lane.registry.get(load_arg)
            timeout = float(entry.load_timeout_s if entry else lane.cfg.load_timeout_s) + 60.0
        elif server == "vllm":
            entry = lane.registry.get(load_arg)
            base = float(entry.load_timeout_s if entry else self.s.default_vllm_load_timeout_s)
            timeout = base + self.s.vllm_ready_grace_s + 30.0
        else:
            timeout = 600.0
        ok, err = await self._wait_job(job.id, timeout)
        if not ok:
            retry = await self._retry_elsewhere(request, model, err, lane,
                                                sub_path, body, is_chat=False)
            if retry is not None:
                return retry
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"failed to load '{model}': {err}",
                                    "type": "server_error", "code": "model_load_failed"}},
            )
        if _spark_shaped(lane):
            # Same slot re-route as the chat path — an embedding answered by the
            # wrong slot's model would not even error, just embed wrongly.
            backend = self._route(lane, server, model, await lane.status(), backend)
        resp = await self.forward(backend, sub_path, body, request.headers,
                                  extra_headers=self._unit_headers(lane))
        lane.touch(model=model)
        return resp

    async def models(self, request: Request) -> dict:
        header = request.headers.get("x-llm-lane")
        # With auto-placement, the natural catalog for a client that names no unit
        # is the UNION across the fleet — one entry per id, first unit's label
        # (auto-routing makes any one entry correct). An explicit header keeps the
        # per-unit list (opencode's providers depend on it).
        if self.placer is not None and self.s.auto_place_enabled and wants_auto(header):
            lanes = [u for u in self.orch.units.values() if u.cfg.enabled]
        else:
            lanes = [self.lane(header or "primary")]
        # Track the backend each id came from (the gateway's own source of truth), so
        # the display `name` is tagged from that — not guessed from the id convention.
        tagged: list[tuple[str, str]] = []  # (id, backend label)
        for lane in lanes:
            if _spark_shaped(lane):
                # A group lists its CLUSTER catalog, so a client can name the
                # multi-node model and have auto-placement route or cold-start it.
                for se in lane.registry.entries():
                    tagged.append((se.served_name or se.alias, f"Spark {lane.cfg.name}"))
            else:
                if lane.cfg.vllm_enabled:      # Ollama-only lanes list no aliases
                    for e in lane.registry.entries():
                        tagged.append((e.served_name or e.alias, "vLLM"))
                try:
                    for m in await lane.ollama.list_models():
                        if m.name:
                            tagged.append((m.name, "Ollama"))
                except Exception:
                    pass
        seen: set[str] = set()
        data = []
        for mid, label in tagged:
            if mid and mid not in seen:
                seen.add(mid)
                # `id` stays the canonical handle the resolver/opencode match on; `name`
                # is purely additive (Open WebUI uses it for the picker label).
                data.append({"id": mid, "name": f"{mid}  ({label})",
                             "object": "model", "owned_by": "llmconfig"})
        return {"object": "list", "data": data}


def build_gateway_router(gateway: OpenAIGateway) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai-gateway"])

    @router.get("/models")
    async def v1_models(request: Request) -> dict:
        return await gateway.models(request)

    @router.post("/chat/completions")
    async def v1_chat_completions(request: Request):
        return await gateway.handle_completion(request, "/chat/completions", is_chat=True)

    @router.post("/completions")
    async def v1_completions(request: Request):
        return await gateway.handle_completion(request, "/completions", is_chat=False)

    @router.post("/embeddings")
    async def v1_embeddings(request: Request):
        return await gateway.handle_pooling(request, "/embeddings")

    # vLLM serves rerank at /v1/rerank; /v1/score is the same scorer under the name
    # some clients (and the Jina/Cohere-style SDKs) expect. Both are plain passthrough.
    @router.post("/rerank")
    async def v1_rerank(request: Request):
        return await gateway.handle_pooling(request, "/rerank")

    @router.post("/score")
    async def v1_score(request: Request):
        return await gateway.handle_pooling(request, "/score")

    return router
