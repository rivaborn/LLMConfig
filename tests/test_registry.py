from llmconfig.registry import Registry
from llmconfig.schemas import VllmAliasEntry


def test_seed_and_crud(tmp_path):
    path = tmp_path / "vllm_models.yaml"
    reg = Registry(path)
    assert path.exists(), "registry should seed from the packaged default on first load"

    aliases = {e.alias for e in reg.entries()}
    assert {"smoke", "coder30-awq", "devstral"} <= aliases
    assert reg.served_name("coder30-awq") == "qwen3-coder-30b"

    reg.upsert(VllmAliasEntry(alias="custom", served_name="custom-x", managed_by="registry"))
    assert Registry(path).get("custom").served_name == "custom-x"

    assert reg.remove("custom") is True
    assert Registry(path).get("custom") is None


def test_blocked_status_present(tmp_path):
    # The status field carries "blocked" (the loader refuses such aliases). Use a
    # synthetic entry so the test doesn't depend on which catalog aliases are blocked.
    path = tmp_path / "r.yaml"
    reg = Registry(path)
    reg.upsert(VllmAliasEntry(alias="zzz-blocked", served_name="zzz",
                              status="blocked", managed_by="serve.sh"))
    blocked = {e.alias for e in Registry(path).entries() if e.status == "blocked"}
    assert "zzz-blocked" in blocked
