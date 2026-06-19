# Deploying LLMConfig on .40 (`Alien-3070-TI`)

The app runs **Windows-native** on the LLM box and reaches into WSL2 for vLLM.

## 1. Get the code + venv (Windows side)
```powershell
git clone https://github.com/rivaborn/LLMConfig C:\Coding\rivaborn\LLMConfig
cd C:\Coding\rivaborn\LLMConfig
python -m venv .venv
.\.venv\Scripts\pip install -e .
copy .env.example .env      # edit if any defaults differ from this box
```

## 2. Install the vLLM systemd-user unit (WSL side)
```bash
# inside WSL: wsl -d Ubuntu-24.04 -u folar
mkdir -p ~/.config/systemd/user
cp /mnt/c/Coding/rivaborn/LLMConfig/deploy/vllm@.service ~/.config/systemd/user/
systemctl --user daemon-reload
# lingering should already be enabled (the vllm-relay unit needs it):
loginctl enable-linger folar
```
> Edit `ExecStart` in `vllm@.service` if your `serve.sh` is not at `/home/folar/vllm/serve.sh`.

## 3. Verify the box matches expectations
```powershell
.\.venv\Scripts\llmconfig doctor --local
```
Fix any `FAIL`/`WARN` (serve.sh path, the `vllm@` unit, `systemctl --user`, service-control elevation, the 3090 UUID) before relying on swaps.

## 4. Run it
Foreground:
```powershell
.\.venv\Scripts\llmconfig serve            # or: .\.venv\Scripts\python -m uvicorn llmconfig.main:app --host 0.0.0.0 --port 11430
```
Always-on (elevated — needed so it can Restart-Service ollama) + firewall rule:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\install-service.ps1
```

UI: `http://192.168.1.40:11430/` · API docs: `…/docs`

## 5. (Optional) Companion lane — the RTX 3070 Ti

A second, independent lane that runs its own small model on the 3070 Ti (8 GB) while
the 3090 keeps doing its own thing. Each lane arbitrates Ollama⇄vLLM on its own card;
they never evict each other.

**a. Second Ollama instance (Windows side)** — pinned to the 3070 Ti, on port 11435,
sharing the primary model store:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\install-companion.ps1
```

**b. Enable the lane** in `.env`, then verify:
```powershell
# .env:  COMPANION_ENABLED=true
.\.venv\Scripts\llmconfig doctor --local      # companion.gpu + companion.ollama.* should pass
```

**c. (Optional) Companion vLLM (WSL side)** — only if you want vLLM (not just Ollama)
on the 3070 Ti. Mirror the primary vLLM setup with a 3070 Ti-pinned variant:
```bash
# inside WSL: wsl -d Ubuntu-24.04 -u folar
# 1) serve-companion.sh — like serve.sh but pins the 3070 Ti and uses a lower
#    --gpu-memory-utilization + small (≤8 GB) models; serves on an internal port (e.g. 11439).
#    Its alias table must match llmconfig/data/vllm_models_companion.default.yaml.
# 2) a 2nd socat relay: 127.0.0.1:11438  →  the companion vLLM internal port (11439).
# 3) install the unit:
cp /mnt/c/Coding/rivaborn/LLMConfig/deploy/vllm-companion@.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

**d. Pick what runs on it.** Load on demand from the UI/CLI, or set a sticky default
that auto-loads on startup:
```bash
llmconfig load --lane companion ollama qwen3:4b      # load now
llmconfig companion-default ollama qwen3:4b          # auto-load on every startup
llmconfig status                                     # shows both lanes
```

> **GPU pinning (verified live on `.40`):**
> - **Ollama needs a device *index*, not a UUID.** Ollama's `CUDA_VISIBLE_DEVICES`
>   does *not* resolve `GPU-<uuid>` — given a UUID it discovers no GPU and silently
>   runs on CPU (`library=cpu`, `total_vram=0 B` in its log). `install-companion.ps1`
>   keeps the UUID as the source of truth but translates it to an index under
>   `CUDA_DEVICE_ORDER=PCI_BUS_ID` (so indices match `nvidia-smi`). vLLM/PyTorch *do*
>   accept the UUID, so `vllm-companion@.service` pins by UUID.
> - **This box already pins to the 3090 via a User-scope `CUDA_VISIBLE_DEVICES=1`**
>   (PCI_BUS_ID order → index 1 = the 3090). That's the de-facto "primary pinned to
>   3090". The companion NSSM service runs as LocalSystem (does *not* inherit the User
>   env) and sets the 3070 Ti index explicitly, so it lands on the right card. Never
>   widen that pin to a Machine/global var — it would leak into the companion.
> - After install, confirm offload is real: load a small model on the companion and
>   check `nvidia-smi` shows its VRAM rise on the 3070 Ti (and the service log does
>   *not* say `library=cpu`).

## Notes
- If `LLMCONFIG_API_KEY` is set in `.env`, write ops require the `X-API-Key` header (the UI has a field; the CLI reads `$LLMCONFIG_API_KEY`).
- The app must run with rights to control the `ollama` service — NSSM's LocalSystem or the elevated scheduled task covers this; a plain user shell may hit "access denied" on `Restart-Service`.
- vLLM is reached at `127.0.0.1:11437` (the socat relay) — never `localhost` (IPv4 happy-eyeballs).
- **WSL persistence:** WSL2 shuts the distro down ~seconds after the last `wsl.exe`
  call exits, which would kill a just-loaded vLLM model (and the relay). The app
  handles this itself — a vLLM load starts a `wsl.exe … sleep infinity` keepalive
  that holds the distro open until the app stops. No extra step is needed. (If the
  app is *killed* rather than stopped gracefully, the keepalive is orphaned and
  keeps the distro up harmlessly; `wsl --shutdown` clears it.)
- `serve.sh` is invoked as `bash serve.sh <alias>` by the `vllm@` unit, so it does
  not strictly need its `+x` bit — but `doctor` checks `test -x`, so keep it
  executable (`chmod +x ~/vllm/serve.sh`) to keep the check green.
