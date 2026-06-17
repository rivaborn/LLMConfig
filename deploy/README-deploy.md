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

## Notes
- If `LLMCONFIG_API_KEY` is set in `.env`, write ops require the `X-API-Key` header (the UI has a field; the CLI reads `$LLMCONFIG_API_KEY`).
- The app must run with rights to control the `ollama` service — NSSM's LocalSystem or the elevated scheduled task covers this; a plain user shell may hit "access denied" on `Restart-Service`.
- vLLM is reached at `127.0.0.1:11437` (the socat relay) — never `localhost` (IPv4 happy-eyeballs).
