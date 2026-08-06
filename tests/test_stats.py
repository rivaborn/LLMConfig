"""Usage stats (`llmconfig/stats.py`) — request buckets, eviction events, the
best-effort persistence contract, and the /api/stats endpoints' shapes."""
import sqlite3
import time

from llmconfig.config import Settings
from llmconfig.stats import UsageStats


def make(tmp_path, **over):
    s = Settings(_env_file=None, stats_db_path=tmp_path / "stats.db", **over)
    return UsageStats(s)


# --------------------------------------------------------------------------- #
# Recording + reads
# --------------------------------------------------------------------------- #
def test_requests_aggregate_into_hourly_buckets(tmp_path):
    st = make(tmp_path)
    for _ in range(3):
        st.note_request("primary", "qwen3:4b", "interactive")
    st.note_request("spark1", "qwen3:4b", "batch")
    st.note_request("spark1", "gemma", "")
    out = st.models(days=1)
    by_model = {m["model"]: m for m in out}
    assert by_model["qwen3:4b"]["requests"] == 4
    assert by_model["qwen3:4b"]["units"] == {"primary": 3, "spark1": 1}
    assert by_model["qwen3:4b"]["workloads"] == {"interactive": 3, "batch": 1}
    assert by_model["gemma"]["workloads"] == {"unclassified": 1}
    assert out[0]["model"] == "qwen3:4b", "most-requested first"
    assert by_model["qwen3:4b"]["share_pct"] == 80.0
    # Hour-resolution, but never in the future: the bucket a live request lands
    # in ends up to 59 min from now, and reporting that as "last used" would
    # read as a timestamp that hasn't happened yet.
    assert by_model["qwen3:4b"]["last_used_ts"] <= time.time() + 1e-6
    assert by_model["qwen3:4b"]["last_used_ts"] > time.time() - 3600


def test_evictions_record_reason_span_and_holder(tmp_path):
    st = make(tmp_path)
    st.note_eviction(unit="spark1", model="served-1", alias="m1",
                     reason="idle_preempt", evicted_by="opencode",
                     incoming_model="m2", incoming_priority=60, holder="batchapp")
    st.note_eviction(unit="spark1_spark2", model="ds4-served", alias="ds4",
                     span=2, reason="group_preempt", evicted_by="opencode")
    evs = st.evictions(limit=10)
    assert len(evs) == 2 and evs[0]["reason"] == "group_preempt", "newest first"
    assert evs[0]["span"] == 2
    assert evs[1]["holder"] == "batchapp" and evs[1]["incoming_priority"] == 60
    # Eviction counts fold into the per-model view.
    counts = {m["model"]: m["evictions"] for m in st.models(days=1)}
    assert counts["m1"] == {"idle_preempt": 1}
    assert counts["ds4"] == {"group_preempt": 1}


def test_disabled_stats_record_nothing(tmp_path):
    st = make(tmp_path, stats_enabled=False)
    st.note_request("primary", "qwen3:4b")
    st.note_eviction(unit="primary", model="x", reason="idle_reaper")
    assert st.models(days=1) == [] and st.evictions() == []


# --------------------------------------------------------------------------- #
# Persistence — best-effort SQLite, Monitor's contract
# --------------------------------------------------------------------------- #
async def test_flush_persists_and_a_fresh_instance_reads_it_back(tmp_path):
    st = make(tmp_path)
    st.start()
    st.note_request("primary", "qwen3:4b", "interactive")
    st.note_eviction(unit="primary", model="old", reason="displaced_by_load")
    await st.stop()          # final flush + close

    fresh = make(tmp_path)
    fresh.start()
    try:
        models = fresh.models(days=1)
        assert [m["model"] for m in models] == ["qwen3:4b", "old"]
        assert fresh.evictions()[0]["model"] == "old"
    finally:
        await fresh.stop()


async def test_db_failure_disables_persistence_but_collection_continues(tmp_path):
    st = make(tmp_path)
    st.start()
    st.note_request("primary", "qwen3:4b")
    # Break the DB under it: the next flush must degrade, not raise.
    st._db.close()
    st._flush()
    assert st._db is None, "persistence disabled after the failure"
    st.note_request("primary", "qwen3:4b")
    st.note_eviction(unit="primary", model="x", reason="idle_reaper")
    by_model = {m["model"]: m for m in st.models(days=1)}
    assert by_model["qwen3:4b"]["requests"] == 2, "in-memory collection continues"
    assert st.evictions()[0]["model"] == "x"
    await st.stop()


async def test_prune_drops_rows_past_retention(tmp_path):
    st = make(tmp_path, stats_retention_days=1)
    st.start()
    old_hour = int((time.time() - 3 * 86400) // 3600) * 3600
    with st._db_lock:
        st._db.execute("INSERT INTO request_buckets VALUES (?,?,?,?,?)",
                       (old_hour, "primary", "ancient", "", 5))
        st._db.execute("INSERT INTO eviction_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (time.time() - 3 * 86400, "primary", "ancient", "ancient",
                        1, "idle_reaper", "", "", None, ""))
        st._db.commit()
    st._last_prune = 0.0
    st.note_request("primary", "current")
    st._flush()
    with st._db_lock:
        assert st._db.execute("SELECT COUNT(*) FROM request_buckets "
                              "WHERE model='ancient'").fetchone()[0] == 0
        assert st._db.execute("SELECT COUNT(*) FROM eviction_events").fetchone()[0] == 0
    assert st.models(days=1)[0]["model"] == "current"
    await st.stop()


async def test_note_request_touches_no_sqlite(tmp_path):
    """The hot path is sync and I/O-free — only the flush task writes."""
    st = make(tmp_path)
    st.start()

    class Exploder:
        def __getattr__(self, name):
            raise AssertionError("sqlite touched on the hot path")
    real_db = st._db
    st._db = Exploder()
    st.note_request("primary", "qwen3:4b", "interactive")
    st.note_eviction(unit="primary", model="x", reason="idle_reaper")
    st._db = real_db
    await st.stop()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
async def test_api_stats_endpoints_shapes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LLMCONFIG_DATA_DIR", str(tmp_path))
    from llmconfig import config as config_mod
    from llmconfig import main as main_mod
    monkeypatch.setattr(config_mod, "get_settings",
                        lambda: Settings(_env_file=None,
                                         stats_db_path=tmp_path / "stats.db",
                                         registry_path=tmp_path / "reg.yaml",
                                         monitor_enabled=False))
    monkeypatch.setattr(main_mod, "get_settings", config_mod.get_settings)
    app = main_mod.create_app()
    app.state.stats.note_request("primary", "qwen3:4b", "interactive")
    app.state.stats.note_eviction(unit="primary", model="old",
                                  reason="displaced_by_load", evicted_by="api")
    with TestClient(app) as c:
        models = c.get("/api/stats/models?days=7").json()
        evs = c.get("/api/stats/evictions?limit=5").json()
    assert models["days"] == 7
    assert models["models"][0]["model"] == "qwen3:4b"
    assert evs["evictions"][0]["reason"] == "displaced_by_load"
