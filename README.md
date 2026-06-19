# LLMConfig

A small control plane that lets you pick **which model runs on the GPU** and on
**which server** — Ollama or vLLM — on a single-GPU box, and guarantees the
chosen model is the *only* thing in VRAM so it never spills to system RAM
prematurely.

Built for `192.168.1.40` (`Alien-3070-TI`, Windows 11, **RTX 3090 24 GB**), which
runs two inference servers that share the one card:

| | Ollama | vLLM |
|---|---|---|
| Runs in | Windows 11 native | WSL2 Ubuntu 24.04 |
| Reach | `127.0.0.1:11434` | `127.0.0.1:11437` (socat relay) |
| Swap | on-demand via REST | one model/process — `serve.sh <alias>` |

The 24 GB card holds **one model at a time across both servers**. LLMConfig
automates the arbitration: to load a model it evicts the other server and any
other Ollama models, **waits until `nvidia-smi` confirms VRAM is actually freed**,
then loads the target — so it packs 100 % of VRAM before any CPU spill.

## What it does
- **Pick + load** a model on Ollama or vLLM, GPU-arbitrated, via Web UI / REST / CLI.
- **Query** available models (Ollama tags + vLLM alias catalog), what's currently
  loaded, on which server, and live VRAM.
- **Verify** packing: reports on-GPU vs on-CPU bytes and flags premature spill;
  optional `--max-pack` pushes `num_gpu` to fill VRAM first.
- **Manage** models: pull/delete Ollama models, edit the vLLM alias registry,
  trigger HF downloads.
- **`doctor`**: read-only recon that checks every on-box assumption.
- **Companion lane** (optional): drive a second GPU (e.g. the RTX 3070 Ti) as an
  independent lane that runs its own model — so the 3090 can serve a big vLLM model
  while the 3070 Ti simultaneously serves a small Ollama/vLLM model, with no
  cross-lane eviction. Off by default; enable with `COMPANION_ENABLED=1`
  (setup: [`deploy/README-deploy.md`](deploy/README-deploy.md)).

## Quickstart (on the box)
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
copy .env.example .env
.\.venv\Scripts\llmconfig doctor --local   # verify the box matches config
.\.venv\Scripts\llmconfig serve            # http://<box>:11430/
```
```bash
llmconfig status
llmconfig load vllm coder30-awq
llmconfig load ollama qwen3-coder:30b
llmconfig unload
```
Full deployment (systemd unit, Windows service, firewall): see
[`deploy/README-deploy.md`](deploy/README-deploy.md).

## How it works
```
            Web UI / CLI / REST
                    │
              FastAPI (Windows-native on .40)
                    │
        ┌───────────┴────────────┐
   OllamaBackend            VllmBackend
   httpx :11434             httpx relay :11437  (status)
   PowerShell svc ctl       wsl.exe → serve.sh / systemctl --user (lifecycle)
        └───────────┬────────────┘
              Orchestrator  ── nvidia-smi (3090 by UUID) ── eviction-wait gate
```
Layout: `llmconfig/{config,wsl,winsvc,gpu,proc,registry,jobs,orchestrator,doctor,main,cli}.py`,
`llmconfig/backends/{ollama,vllm}.py`, `llmconfig/web/`, `llmconfig/data/vllm_models.default.yaml`.

## Configuration
All box-specific values live in `.env` (see `.env.example`): ports, the Ollama
service name, the vLLM relay URL + serve.sh path + systemd unit, the WSL distro/user,
the **3090 UUID**, VRAM thresholds, optional `LLMCONFIG_API_KEY`, and `HF_TOKEN`.

## REST API
`GET /api/status` (incl. per-lane `lanes[]`) · `GET /api/models?lane=` · `GET /api/gpu?lane=` ·
`GET /api/lanes` · `GET/PUT /api/lanes/{id}/default` · `GET /api/doctor` ·
`POST /api/load {server,model,lane?,force?,max_pack?}` → job · `POST /api/unload {server?,lane?}` ·
`GET /api/jobs/{id}` · `POST /api/ollama/pull` · `DELETE /api/ollama/{name}` ·
`GET/POST/PUT/DELETE /api/vllm/aliases?lane=` · `POST /api/vllm/download`. Interactive docs at `/docs`.

## Status
**Live-verified on `.40` (2026-06-17).** `doctor --local` is green; both paths
were exercised end-to-end on the RTX 3090: Ollama load/unload, and a vLLM load
(`smoke`) that evicted everything, packed VRAM, served real completions through
the relay, and unloaded back to 0 MiB.

One thing live testing caught that the docs didn't: **WSL2 idle-shuts-down the
whole distro a few seconds after the last `wsl.exe` call returns**, which killed
the `vllm@` unit (and the socat relay) seconds after a load — even with
lingering enabled. LLMConfig now holds the distro open for its lifetime via a
`WslKeepalive` (`wsl.exe … sleep infinity`) that a vLLM load starts automatically
and the app releases on graceful shutdown. vLLM control reuses the existing
`serve.sh` + relay; it does not reimplement them.
```
pip install -e ".[dev]" && pytest    # unit tests (registry, gpu parsing, orchestrator)
```
