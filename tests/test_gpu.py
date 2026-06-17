import llmconfig.gpu as gpu
from llmconfig.config import Settings
from llmconfig.proc import CmdResult

UUID = "GPU-739bece9-8298-7993-f7dd-c8d86cb541f9"


async def test_query_gpu_matches_by_uuid(monkeypatch):
    async def fake_run_argv(argv, timeout=15.0):
        joined = " ".join(argv)
        if "--query-gpu" in joined:
            # 3090 is the *second* line — index would be wrong, UUID is right
            return CmdResult(0, f"GPU-other-3070ti, 8192, 50, 8142\n{UUID}, 24576, 1234, 23342\n", "")
        if "--query-compute-apps" in joined:
            return CmdResult(0, "4242, 1200, ollama.exe\n", "")
        return CmdResult(127, "", "no")

    monkeypatch.setattr(gpu, "run_argv", fake_run_argv)
    info = await gpu.query_gpu(Settings(gpu_uuid=UUID))

    assert info.found
    assert info.total_mb == 24576 and info.used_mb == 1234 and info.free_mb == 23342
    assert info.utilization_pct == round(100 * 1234 / 24576, 1)
    assert info.is_free(1500) is True  # used 1234 <= baseline 1500
    assert info.is_free(1000) is False
    assert info.processes[0].name == "ollama.exe" and info.processes[0].used_mb == 1200


async def test_query_gpu_missing_uuid(monkeypatch):
    async def fake_run_argv(argv, timeout=15.0):
        if "--query-gpu" in " ".join(argv):
            return CmdResult(0, "GPU-something-else, 8192, 50, 8142\n", "")
        return CmdResult(0, "", "")

    monkeypatch.setattr(gpu, "run_argv", fake_run_argv)
    info = await gpu.query_gpu(Settings(gpu_uuid=UUID))
    assert not info.found
    assert "not present" in info.error
