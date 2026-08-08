"""Usage stats — request/eviction accounting behind `/api/stats/*`.

RECORD-ONLY: which models are actually used, how often, and how often
preemption/eviction fires — the data for tuning eviction preferences (and, one
day, for feeding a least-used tie-break into placement's victim ranking, which
deliberately does NOT read this yet). Two kinds of record:

* **Requests** — one note per gateway request, aggregated into HOURLY buckets
  per (unit, model, workload class). Raw per-request rows would be write-heavy
  and answer no question the buckets can't.
* **Evictions** — raw events (they are rare): who was displaced, from where,
  spanning how many nodes, by whom, and WHY (`reason` distinguishes an idle
  preemption from an active priority preemption from the reaper, etc.).

Persistence follows the Monitor's contract exactly: best-effort SQLite
(`stats_db_path`) — a DB failure disables persistence and collection continues
in-memory; history then simply doesn't survive a restart. The hot path is sync
and I/O-free (dict/deque mutation only): a background task flushes dirty
buckets and pending events every `stats_flush_interval_s` and prunes both
tables past `stats_retention_days`. No sqlite call ever sits between a final
sync lease check and an unload (invariants 11/17) — recorders are called AFTER
the stop/unload has been issued, and even then only buffer.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .config import Settings

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_buckets (
    hour_ts  INTEGER NOT NULL,
    unit     TEXT NOT NULL,
    model    TEXT NOT NULL,
    workload TEXT NOT NULL,
    client   TEXT NOT NULL DEFAULT '',
    n        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hour_ts, unit, model, workload, client)
);
CREATE TABLE IF NOT EXISTS eviction_events (
    ts REAL, unit TEXT, model TEXT, alias TEXT, span INTEGER,
    reason TEXT, evicted_by TEXT, incoming_model TEXT,
    incoming_priority INTEGER, holder TEXT
);
CREATE INDEX IF NOT EXISTS ix_eviction_events_ts ON eviction_events(ts);
CREATE TABLE IF NOT EXISTS clients (
    app        TEXT NOT NULL,
    ip         TEXT NOT NULL,
    fn         TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL,
    n          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (app, ip, fn)
);
"""

# In-memory fallback bound: enough hours × models to cover the retention window
# for this fleet without the dict ever mattering memory-wise.
_MAX_EVENTS_MEMORY = 500


class UsageStats:
    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = bool(settings.stats_enabled)
        self.retention_s = max(1, int(settings.stats_retention_days)) * 86400
        # Unflushed request deltas: (hour_ts, unit, model, workload, client) -> n.
        self._pending: dict[tuple[int, str, str, str, str], int] = {}
        # Unflushed eviction rows + a bounded recent window for DB-less reads.
        self._events_pending: list[tuple] = []
        self._events_recent: deque[dict] = deque(maxlen=_MAX_EVENTS_MEMORY)
        # In-memory bucket fallback (kept even when the DB works — merged into
        # reads so the current interval is never invisible).
        self._buckets_memory: dict[tuple[int, str, str, str, str], int] = {}
        # Clients registry, same two-tier pattern as the buckets:
        # (app, ip, fn) -> [first_seen, last_seen, n]. `_clients_pending` holds
        # unflushed deltas; `_clients_memory` is the full DB-less fallback.
        self._clients_pending: dict[tuple[str, str, str], list] = {}
        self._clients_memory: dict[tuple[str, str, str], list] = {}
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self._stop_evt = asyncio.Event()
        self._last_prune = 0.0

    # ---- lifecycle ----
    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        try:
            path = Path(self.s.stats_db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(path), check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            self._migrate_buckets(db)
            db.executescript(_SCHEMA)
            db.commit()
            self._db = db
        except Exception as e:  # noqa: BLE001 — never let persistence block startup
            log.warning("stats persistence disabled (%s): %s", type(e).__name__, e)
            self._db = None
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._loop(), name="stats-flush")

    @staticmethod
    def _migrate_buckets(db: sqlite3.Connection) -> None:
        """v1 request_buckets (no `client` column) -> v2, preserving history.

        SQLite can't ALTER a PRIMARY KEY, so: rename aside, let _SCHEMA create
        the v2 table, copy rows across with client='', drop the old table.
        Runs BEFORE executescript so CREATE IF NOT EXISTS sees no stale v1
        table. Telemetry-sized (retention is ~30 days of hourly buckets); on
        any failure the caller's except degrades to in-memory, per this file's
        contract — history is nice-to-have, startup is not negotiable.
        """
        cols = [r[1] for r in db.execute("PRAGMA table_info(request_buckets)")]
        if not cols or "client" in cols:
            return                          # no table yet, or already v2
        db.execute("ALTER TABLE request_buckets RENAME TO request_buckets_v1")
        db.executescript(_SCHEMA)
        db.execute(
            "INSERT INTO request_buckets(hour_ts,unit,model,workload,client,n) "
            "SELECT hour_ts,unit,model,workload,'',n FROM request_buckets_v1")
        db.execute("DROP TABLE request_buckets_v1")
        db.commit()
        log.info("stats: migrated request_buckets to v2 (per-client key)")

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        self._flush()
        if self._db is not None:
            try:
                with self._db_lock:
                    self._db.commit()
                    self._db.close()
            except Exception:  # noqa: BLE001
                pass
            self._db = None

    async def _loop(self) -> None:
        interval = max(5.0, float(self.s.stats_flush_interval_s))
        while not self._stop_evt.is_set():
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if self._stop_evt.is_set():
                break
            try:
                await asyncio.to_thread(self._flush)
            except Exception as e:  # noqa: BLE001 — a flush hiccup must never kill the loop
                log.debug("stats flush failed: %s", e)

    # ---- recorders (sync, no I/O — safe anywhere) ----
    def note_request(self, unit: str, model: str, workload: str = "",
                     client: str = "", ip: str = "", fn: str = "") -> None:
        """One gateway request routed to (unit, model). Called ONCE per request
        (not per touch — touch fires again after generation).

        `client` is the resolved app name (X-LLM-App, else the hold name, else
        the lease holder, else "unknown"); `ip` the peer address captured
        server-side; `fn` the free-text X-LLM-Function. The bucket key carries
        `client@ip`; `fn` is registry-only (too free-form for bucket keys)."""
        if not self.enabled or not model:
            return
        now = time.time()
        who = f"{client or 'unknown'}@{ip}" if ip else (client or "")
        key = (int(now // 3600) * 3600, unit, model, workload or "", who)
        self._pending[key] = self._pending.get(key, 0) + 1
        self._buckets_memory[key] = self._buckets_memory.get(key, 0) + 1
        if ip:                              # registry: who is using the fleet
            ck = (client or "unknown", ip, fn or "")
            for d in (self._clients_pending, self._clients_memory):
                row = d.get(ck)
                if row is None:
                    d[ck] = [now, now, 1]
                else:
                    row[1] = now
                    row[2] += 1

    def note_eviction(self, *, unit: str, model: str, alias: str = "",
                      span: int = 1, reason: str, evicted_by: str = "",
                      incoming_model: str = "",
                      incoming_priority: Optional[int] = None,
                      holder: str = "") -> None:
        """One displaced model. `reason` is the path that displaced it:
        idle_preempt | active_preempt | displaced_idle | displaced_by_load |
        group_preempt | idle_reaper | lease_free."""
        if not self.enabled:
            return
        ts = time.time()
        row = (ts, unit, model, alias or model, int(span), reason, evicted_by,
               incoming_model, incoming_priority, holder)
        self._events_pending.append(row)
        self._events_recent.append({
            "ts": round(ts, 1), "unit": unit, "model": model,
            "alias": alias or model, "span": int(span), "reason": reason,
            "evicted_by": evicted_by, "incoming_model": incoming_model,
            "incoming_priority": incoming_priority, "holder": holder,
        })

    # ---- flush / prune (worker thread) ----
    def _flush(self) -> None:
        pending, self._pending = self._pending, {}
        events, self._events_pending = self._events_pending, []
        clients, self._clients_pending = self._clients_pending, {}
        # Prune the in-memory fallback buckets past retention regardless of DB.
        cutoff_hour = int((time.time() - self.retention_s) // 3600) * 3600
        for k in [k for k in self._buckets_memory if k[0] < cutoff_hour]:
            self._buckets_memory.pop(k, None)
        for k in [k for k, v in self._clients_memory.items()
                  if v[1] < time.time() - self.retention_s]:
            self._clients_memory.pop(k, None)
        if self._db is None:
            return
        try:
            with self._db_lock:
                for (hour_ts, unit, model, workload, client), n in pending.items():
                    self._db.execute(
                        "INSERT INTO request_buckets"
                        "(hour_ts,unit,model,workload,client,n) "
                        "VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(hour_ts,unit,model,workload,client) "
                        "DO UPDATE SET n = n + excluded.n",
                        (hour_ts, unit, model, workload, client, n))
                for (app, ip, fn), (first, last, dn) in clients.items():
                    self._db.execute(
                        "INSERT INTO clients(app,ip,fn,first_seen,last_seen,n) "
                        "VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(app,ip,fn) DO UPDATE SET "
                        "last_seen = MAX(last_seen, excluded.last_seen), "
                        "n = n + excluded.n",
                        (app, ip, fn, first, last, dn))
                if events:
                    self._db.executemany(
                        "INSERT INTO eviction_events(ts,unit,model,alias,span,"
                        "reason,evicted_by,incoming_model,incoming_priority,holder) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)", events)
                now = time.time()
                if now - self._last_prune > 86400:
                    self._db.execute(
                        "DELETE FROM request_buckets WHERE hour_ts < ?",
                        (cutoff_hour,))
                    self._db.execute(
                        "DELETE FROM eviction_events WHERE ts < ?",
                        (now - self.retention_s,))
                    self._db.execute(
                        "DELETE FROM clients WHERE last_seen < ?",
                        (now - self.retention_s,))
                    self._last_prune = now
                self._db.commit()
        except Exception as e:  # noqa: BLE001 — degrade to in-memory, keep collecting
            log.warning("stats persist failed (%s) — continuing in-memory: %s",
                        type(e).__name__, e)
            self._db = None

    # ---- reads (endpoint-facing; DB + unflushed buffer merged) ----
    def models(self, days: int = 30) -> list[dict]:
        """Per-model usage over the trailing window, most-requested first."""
        now = time.time()
        since = now - max(1, days) * 86400
        since_hour = int(since // 3600) * 3600
        agg: dict[str, dict] = {}

        def add(model: str, unit: str, workload: str, n: int, hour_ts: int) -> None:
            m = agg.setdefault(model, {
                "model": model, "requests": 0, "last_used_ts": None,
                "units": {}, "workloads": {}, "evictions": {},
            })
            m["requests"] += n
            m["units"][unit] = m["units"].get(unit, 0) + n
            wl = workload or "unclassified"
            m["workloads"][wl] = m["workloads"].get(wl, 0) + n
            # Buckets are hourly, so the best "last used" this can offer is the
            # END of the newest bucket holding a request — clamped to now, or
            # the model in use right now would report up to 59 min in the FUTURE.
            end = min(hour_ts + 3600, now)
            if m["last_used_ts"] is None or end > m["last_used_ts"]:
                m["last_used_ts"] = end

        if self._db is not None:
            try:
                with self._db_lock:
                    rows = self._db.execute(
                        "SELECT hour_ts,unit,model,workload,n FROM request_buckets "
                        "WHERE hour_ts >= ?", (since_hour,)).fetchall()
                for hour_ts, unit, model, workload, n in rows:
                    add(model, unit, workload, n, hour_ts)
            except Exception:  # noqa: BLE001 — reads degrade like writes
                pass
        # Unflushed deltas (the current interval) — DB rows don't have them yet.
        for (hour_ts, unit, model, workload, _client), n in self._pending.items():
            if hour_ts >= since_hour:
                add(model, unit, workload, n, hour_ts)
        if self._db is None:
            # Fallback: the whole in-memory history minus what _pending added.
            for key, n in self._buckets_memory.items():
                hour_ts, unit, model, workload, _client = key
                already = self._pending.get(key, 0)
                if hour_ts >= since_hour and n - already > 0:
                    add(model, unit, workload, n - already, hour_ts)
        for ev in self.evictions(limit=0, since=since):
            m = agg.get(ev["alias"]) or agg.get(ev["model"])
            if m is None:
                m = agg.setdefault(ev["alias"], {
                    "model": ev["alias"], "requests": 0, "last_used_ts": None,
                    "units": {}, "workloads": {}, "evictions": {},
                })
            m["evictions"][ev["reason"]] = m["evictions"].get(ev["reason"], 0) + 1
        total = sum(m["requests"] for m in agg.values()) or 1
        out = sorted(agg.values(), key=lambda m: -m["requests"])
        for m in out:
            m["share_pct"] = round(100.0 * m["requests"] / total, 1)
        return out

    def clients(self, days: int = 30) -> list[dict]:
        """Who is using the fleet: one row per (app, ip, function), most
        recently seen first. DB rows merged with unflushed deltas, so a request
        made a second ago is already visible."""
        since = time.time() - max(1, days) * 86400
        agg: dict[tuple[str, str, str], list] = {}
        if self._db is not None:
            try:
                with self._db_lock:
                    rows = self._db.execute(
                        "SELECT app,ip,fn,first_seen,last_seen,n FROM clients "
                        "WHERE last_seen >= ?", (since,)).fetchall()
                for app, ip, fn, first, last, n in rows:
                    agg[(app, ip, fn)] = [first, last, n]
            except Exception:  # noqa: BLE001 — reads degrade like writes
                pass
            source = self._clients_pending      # DB rows + unflushed deltas
        else:
            source = self._clients_memory       # full in-memory fallback
        for key, (first, last, n) in source.items():
            row = agg.get(key)
            if row is None:
                agg[key] = [first, last, n]
            else:
                row[0] = min(row[0], first)
                row[1] = max(row[1], last)
                row[2] += n
        out = [{"app": k[0], "ip": k[1], "function": k[2],
                "client": f"{k[0]}@{k[1]}",
                "first_seen": round(v[0], 1), "last_seen": round(v[1], 1),
                "requests": v[2]}
               for k, v in agg.items() if v[1] >= since]
        out.sort(key=lambda r: -r["last_seen"])
        return out

    def evictions(self, limit: int = 50, since: float = 0.0) -> list[dict]:
        """Eviction events, newest first. `limit=0` = no cap (internal use)."""
        rows: list[dict] = []
        if self._db is not None:
            try:
                with self._db_lock:
                    q = ("SELECT ts,unit,model,alias,span,reason,evicted_by,"
                         "incoming_model,incoming_priority,holder "
                         "FROM eviction_events WHERE ts >= ? ORDER BY ts DESC")
                    fetched = self._db.execute(q, (since,)).fetchall()
                rows = [{
                    "ts": round(r[0], 1), "unit": r[1], "model": r[2],
                    "alias": r[3], "span": r[4], "reason": r[5],
                    "evicted_by": r[6], "incoming_model": r[7],
                    "incoming_priority": r[8], "holder": r[9],
                } for r in fetched]
            except Exception:  # noqa: BLE001
                rows = []
        # Merge the unflushed tail (kept in _events_recent too, so dedupe by ts).
        seen = {r["ts"] for r in rows}
        for ev in reversed(self._events_recent):
            if ev["ts"] >= since and ev["ts"] not in seen:
                rows.append(ev)
        rows.sort(key=lambda r: -r["ts"])
        return rows[:limit] if limit else rows
