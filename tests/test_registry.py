from llmconfig.registry import Registry
from llmconfig.schemas import VllmAliasEntry


def test_seed_and_crud(tmp_path):
    path = tmp_path / "vllm_models.yaml"
    reg = Registry(path)
    assert path.exists(), "registry should seed from the packaged default on first load"

    aliases = {e.alias for e in reg.entries()}
    assert {"smoke", "coder30-awq", "devstral"} <= aliases
    assert reg.served_name("coder30-awq") == "qwen3-coder-30b"

    # coder30-awq and coder30-fp8 intentionally share a served name
    assert reg.served_name("coder30-fp8") == "qwen3-coder-30b"

    reg.upsert(VllmAliasEntry(alias="custom", served_name="custom-x", managed_by="registry"))
    assert Registry(path).get("custom").served_name == "custom-x"

    assert reg.remove("custom") is True
    assert Registry(path).get("custom") is None


def test_blocked_status_present(tmp_path):
    reg = Registry(tmp_path / "r.yaml")
    blocked = {e.alias for e in reg.entries() if e.status == "blocked"}
    assert {"coder30-fp8", "q36-27b"} <= blocked
