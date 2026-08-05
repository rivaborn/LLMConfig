"""Editing a cookbook state's per-unit rows without staging the fleet.

`snapshot()` can only record what is loaded NOW, so changing which model a state
puts on one node meant loading that whole fleet and re-saving. `edit_units()`
writes the intent directly.

The load-bearing rule: a model spanning several Sparks is NOT settable per-unit.
No special case enforces it — every target is validated against the unit's own
catalog, and by the `SparkModelEntry` invariant a `min_nodes > 1` model lives
only in the cluster catalog, never a per-node one. So it fails to resolve.
"""
import pytest

from llmconfig.cookbook import Cookbook


class FakeCfg:
    def __init__(self, uid, enabled=True):
        self.id = uid
        self.enabled = enabled


class FakeReg:
    def __init__(self, aliases):
        self._a = set(aliases)

    def get(self, alias):
        return object() if alias in self._a else None


class FakeUnit:
    def __init__(self, uid, aliases=(), kind="spark", enabled=True):
        self.cfg = FakeCfg(uid, enabled)
        self.registry = FakeReg(aliases)
        self.kind = kind


class FakeOrch:
    def __init__(self, units):
        self.units = {u.cfg.id: u for u in units}


@pytest.fixture
def book(tmp_path):
    orch = FakeOrch([
        # Per-node catalogs. Note NO multi-node model appears in one — that is
        # the real invariant, mirrored here.
        FakeUnit("spark3", ["qwen35-122b", "qwen35-122b_8_3_26"]),
        FakeUnit("spark4", ["qwen3-vl-reranker-8b", "qwen3-vl-embedding-8b",
                            "qwen35-122b_8_3_26"]),
        FakeUnit("spark1_spark2", [], kind="spark_group"),
        FakeUnit("offline", ["x"], enabled=False),
    ])
    cb = Cookbook.__new__(Cookbook)          # skip __init__'s disk load
    cb.s = None
    cb.orch = orch
    cb.jobs = None
    cb.leases = None
    cb.path = tmp_path / "cookbook.yaml"
    cb._default = "DF4_RAG Daily Driver"
    cb._states = {
        "DF4_RAG Daily Driver": {
            "saved_at": 1.0,
            "units": {"spark3": [{"server": "spark", "model": "qwen35-122b"}],
                      "spark4": [{"server": "spark", "model": "qwen35-122b_8_3_26"}]},
            "groups": {"spark1_spark2": {"model": "deepseek-v4-flash"}},
        }
    }
    return cb


NAME = "DF4_RAG Daily Driver"


def test_edit_replaces_unit_rows(book):
    st = book.edit_units(NAME, {
        "spark3": [{"server": "spark", "model": "qwen35-122b"}],
        "spark4": [{"server": "spark", "model": "qwen3-vl-reranker-8b"},
                   {"server": "spark", "model": "qwen3-vl-embedding-8b"}],
    })
    assert [t["model"] for t in st["units"]["spark4"]] == [
        "qwen3-vl-reranker-8b", "qwen3-vl-embedding-8b"]
    assert st["units"]["spark3"][0]["model"] == "qwen35-122b"


def test_omitted_units_are_preserved(book):
    """Regression, paid for live: a partial body must not drop the rest of the fleet.

    The first real call sent only spark3+spark4 and wiped this state's primary,
    companion, spark1 and spark2 rows. Merge semantics, not replace.
    """
    book._states[NAME]["units"]["primary"] = [{"server": "vllm", "model": "vl32"}]
    st = book.edit_units(NAME, {"spark4": [{"server": "spark", "model": "qwen3-vl-reranker-8b"}]})
    assert st["units"]["primary"] == [{"server": "vllm", "model": "vl32"}]
    assert st["units"]["spark3"][0]["model"] == "qwen35-122b", "untouched unit kept"
    assert [t["model"] for t in st["units"]["spark4"]] == ["qwen3-vl-reranker-8b"]


def test_explicit_empty_still_clears(book):
    """Merge must not cost the ability to pin a unit empty."""
    st = book.edit_units(NAME, {"spark4": []})
    assert st["units"]["spark4"] == []
    assert st["units"]["spark3"][0]["model"] == "qwen35-122b"


def test_groups_are_left_untouched(book):
    """The whole point of the exclusion: editing units must not drop the tp job."""
    st = book.edit_units(NAME, {"spark3": [{"server": "spark", "model": "qwen35-122b"}]})
    assert st["groups"] == {"spark1_spark2": {"model": "deepseek-v4-flash"}}


def test_multi_node_model_is_refused(book):
    """A spanning model is in no per-node catalog, so it cannot be set per-unit."""
    with pytest.raises(ValueError, match="not in its catalog"):
        book.edit_units(NAME, {
            "spark3": [{"server": "spark", "model": "deepseek-v4-flash"}]})


def test_group_unit_id_is_refused(book):
    with pytest.raises(ValueError, match="multi-node group"):
        book.edit_units(NAME, {
            "spark1_spark2": [{"server": "spark", "model": "qwen35-122b"}]})


def test_empty_list_pins_the_unit_empty(book):
    """[] is meaningful — apply frees the unit. It must survive the round trip."""
    st = book.edit_units(NAME, {"spark3": [], "spark4": []})
    assert st["units"]["spark3"] == [] and st["units"]["spark4"] == []


def test_unknown_unit_is_refused(book):
    with pytest.raises(ValueError, match="unknown or disabled unit"):
        book.edit_units(NAME, {"spark9": [{"server": "spark", "model": "qwen35-122b"}]})


def test_disabled_unit_is_refused(book):
    with pytest.raises(ValueError, match="unknown or disabled unit"):
        book.edit_units(NAME, {"offline": [{"server": "spark", "model": "x"}]})


def test_unknown_state_raises_keyerror(book):
    with pytest.raises(KeyError):
        book.edit_units("no-such-state", {"spark3": []})


def test_bad_server_is_refused(book):
    with pytest.raises(ValueError, match="unknown server"):
        book.edit_units(NAME, {"spark3": [{"server": "sglang", "model": "qwen35-122b"}]})


def test_ollama_tags_are_free_form(book):
    """Ollama models load as-is, so they are not catalog-validated (cf. _canonical)."""
    book.orch.units["spark3"].kind = "lane"
    st = book.edit_units(NAME, {"spark3": [{"server": "ollama", "model": "anything:latest"}]})
    assert st["units"]["spark3"][0]["model"] == "anything:latest"


def test_edit_persists_to_disk(book):
    book.edit_units(NAME, {"spark4": [{"server": "spark", "model": "qwen3-vl-reranker-8b"}]})
    assert book.path.exists()
    reread = book.path.read_text(encoding="utf-8")
    assert "qwen3-vl-reranker-8b" in reread
    assert "deepseek-v4-flash" in reread, "groups must persist too"
