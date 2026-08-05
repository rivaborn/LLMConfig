"""Editing a state's MULTI-NODE deployments.

Groups were originally out of the editor's reach on the reading that a spanning
model "must not be added that way". The opposite was meant: a cookbook state is
the only place a whole-fleet layout is written down, so leaving the tp jobs out
made it describe half a fleet.

What must still hold is the physics: a tp job across nodes that are not cabled
together HANGS rather than failing, and a group claims whole nodes.
"""
import pytest

from llmconfig.config import _parse_fabric_link_members, group_id_for
from llmconfig.cookbook import Cookbook


class FakeCfg:
    def __init__(self, uid):
        self.id = uid
        self.enabled = True


class FakeReg:
    def __init__(self, aliases):
        self._a = dict(aliases)

    def get(self, alias):
        return self._a.get(alias)


class FakeEntry:
    def __init__(self, min_nodes=2, max_nodes=4):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes


class FakeUnit:
    def __init__(self, uid, aliases=()):
        self.cfg = FakeCfg(uid)
        self.registry = FakeReg({a: FakeEntry(1, 1) for a in aliases})
        self.kind = "spark"


class FakeOrch:
    def __init__(self, units, cluster):
        self.units = {u.cfg.id: u for u in units}
        self.cluster_registry = FakeReg(cluster)


class FakeSettings:
    spark_fabric_links = "spark1+spark2,spark3+spark4"

    def fabric_link_members(self):
        return _parse_fabric_link_members(self.spark_fabric_links)

    def fabric_links_describe(self):
        return "spark1+spark2 / spark3+spark4"


NAME = "DF4_RAG Daily Driver"


@pytest.fixture
def book(tmp_path):
    orch = FakeOrch(
        [FakeUnit(f"spark{i}", ["qwen35-122b"]) for i in (1, 2, 3, 4)],
        {"deepseek-v4-flash": FakeEntry(2, 4),
         "big-tp4": FakeEntry(4, 4)},
    )
    cb = Cookbook.__new__(Cookbook)
    cb.s = FakeSettings()
    cb.orch = orch
    cb.jobs = cb.leases = None
    cb.path = tmp_path / "cookbook.yaml"
    cb._default = ""
    cb._states = {NAME: {"saved_at": 1.0,
                         "units": {f"spark{i}": [] for i in (1, 2, 3, 4)},
                         "groups": {}}}
    return cb


def test_fabric_member_order_is_preserved():
    """Order picks the HEAD, so a sort would silently re-point the deployment."""
    got = _parse_fabric_link_members("spark1+spark2,spark3+spark4")
    assert got == [("spark1", "spark2"), ("spark3", "spark4")]
    assert got[0][0] == "spark1" and got[1][0] == "spark3"


def test_set_a_group(book):
    st = book.edit_groups(NAME, {"spark3_spark4": {"model": "deepseek-v4-flash"}})
    assert st["groups"]["spark3_spark4"] == {"model": "deepseek-v4-flash"}


def test_clear_a_group_with_none(book):
    book.edit_groups(NAME, {"spark1_spark2": {"model": "deepseek-v4-flash"}})
    st = book.edit_groups(NAME, {"spark1_spark2": None})
    assert "spark1_spark2" not in st["groups"]


def test_uncabled_set_is_refused(book):
    """spark2+spark3 are not linked — a tp job there hangs, which is worse."""
    gid = group_id_for(["spark2", "spark3"])
    with pytest.raises(ValueError, match="not a cabled node set"):
        book.edit_groups(NAME, {gid: {"model": "deepseek-v4-flash"}})


def test_single_node_model_is_refused_for_a_group(book):
    with pytest.raises(ValueError, match="not in the cluster catalog"):
        book.edit_groups(NAME, {"spark1_spark2": {"model": "qwen35-122b"}})


def test_node_count_must_fit_the_model(book):
    """big-tp4 needs four nodes; a pair cannot host it."""
    with pytest.raises(ValueError, match="needs 4"):
        book.edit_groups(NAME, {"spark1_spark2": {"model": "big-tp4"}})


def test_group_conflicts_with_a_member_row(book):
    """A group claims whole nodes — the state would be unapplyable."""
    book._states[NAME]["units"]["spark3"] = [{"server": "spark", "model": "qwen35-122b"}]
    with pytest.raises(ValueError, match="claims\n?.*whole nodes|whole nodes"):
        book.edit_groups(NAME, {"spark3_spark4": {"model": "deepseek-v4-flash"}})


def test_groups_merge_rather_than_replace(book):
    book.edit_groups(NAME, {"spark1_spark2": {"model": "deepseek-v4-flash"}})
    st = book.edit_groups(NAME, {"spark3_spark4": {"model": "deepseek-v4-flash"}})
    assert set(st["groups"]) == {"spark1_spark2", "spark3_spark4"}


def test_editing_groups_leaves_units_alone(book):
    book._states[NAME]["units"]["spark1"] = []
    st = book.edit_groups(NAME, {"spark1_spark2": {"model": "deepseek-v4-flash"}})
    assert set(st["units"]) == {"spark1", "spark2", "spark3", "spark4"}


def test_unknown_state_raises_keyerror(book):
    with pytest.raises(KeyError):
        book.edit_groups("nope", {"spark1_spark2": None})
