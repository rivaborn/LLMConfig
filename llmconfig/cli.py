"""`llmconfig` CLI — a thin client over the REST API (+ `serve` to launch it).

Talks to the app on .40 (default http://127.0.0.1:11430). Point it elsewhere with
--url / $LLMCONFIG_URL (e.g. http://192.168.1.40:11430 over Tailscale).
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
import typer

from .config import get_settings

app = typer.Typer(add_completion=False, help="Control Ollama + vLLM sharing one GPU on .40.")
_STATE: dict[str, Optional[str]] = {"url": None, "api_key": None}


@app.callback()
def _main(
    url: Optional[str] = typer.Option(None, "--url", envvar="LLMCONFIG_URL", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LLMCONFIG_API_KEY", help="X-API-Key"),
) -> None:
    _STATE["url"] = url
    _STATE["api_key"] = api_key


def _client() -> httpx.Client:
    s = get_settings()
    base = _STATE["url"] or s.base_url
    key = _STATE["api_key"] if _STATE["api_key"] is not None else s.llmconfig_api_key
    headers = {"X-API-Key": key} if key else {}
    return httpx.Client(base_url=base, headers=headers, timeout=60.0)


def _bail(e: Exception) -> None:
    s = get_settings()
    base = _STATE["url"] or s.base_url
    typer.secho(f"cannot reach LLMConfig at {base}: {e}", fg="red", err=True)
    raise typer.Exit(2)


def _gib(n: int) -> str:
    return f"{(n or 0) / (1024 ** 3):.1f}G"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def serve() -> None:
    """Run the API + web UI (uvicorn) on this machine."""
    import uvicorn

    s = get_settings()
    typer.echo(f"LLMConfig on http://{s.llmconfig_host}:{s.llmconfig_port}  (UI at {s.base_url}/)")
    uvicorn.run("llmconfig.main:app", host=s.llmconfig_host, port=s.llmconfig_port, log_level="info")


@app.command()
def status() -> None:
    """Show the GPU owner, loaded model, and VRAM."""
    try:
        with _client() as c:
            d = c.get("/api/status").json()
    except httpx.HTTPError as e:
        _bail(e)
    _print_status(d)


@app.command()
def models() -> None:
    """List available Ollama models and vLLM aliases."""
    try:
        with _client() as c:
            d = c.get("/api/models").json()
    except httpx.HTTPError as e:
        _bail(e)
    typer.secho("Ollama:", bold=True)
    for m in d.get("ollama", []):
        mark = "●" if m["loaded"] else " "
        typer.echo(f"  {mark} {m['name']:<28} {_gib(m['size_bytes'])}")
    if d.get("ollama_error"):
        typer.secho(f"  (error: {d['ollama_error']})", fg="yellow")
    typer.secho("vLLM (serve.sh aliases):", bold=True)
    for a in d.get("vllm", []):
        mark = "●" if a["loaded"] else " "
        typer.echo(f"  {mark} {a['alias']:<14} -> {a['served_name']:<20} [{a['status']}] {a['notes'][:40]}")
    if d.get("vllm_error"):
        typer.secho(f"  (error: {d['vllm_error']})", fg="yellow")


@app.command()
def gpu() -> None:
    """Show nvidia-smi state for the configured GPU."""
    try:
        with _client() as c:
            d = c.get("/api/gpu").json()
    except httpx.HTTPError as e:
        _bail(e)
    if not d["found"]:
        typer.secho(f"GPU not found: {d.get('error')}", fg="red")
        raise typer.Exit(1)
    typer.echo(f"{d['uuid']}\n  {d['used_mb']}/{d['total_mb']} MiB used ({d['utilization_pct']}%)  free {d['free_mb']} MiB")
    for p in d.get("processes", []):
        typer.echo(f"    pid {p['pid']:>7}  {p['used_mb']:>6} MiB  {p['name']}")


@app.command()
def load(
    server: str = typer.Argument(..., help="ollama | vllm"),
    model: str = typer.Argument(..., help="Ollama tag or vLLM serve.sh alias"),
    force: bool = typer.Option(False, "--force", help="reload even if already active"),
    max_pack: bool = typer.Option(False, "--max-pack", help="fill VRAM (num_gpu) before spilling (Ollama)"),
) -> None:
    """Load a model on a server, evicting everything else from the GPU first."""
    if server not in ("ollama", "vllm"):
        typer.secho("server must be 'ollama' or 'vllm'", fg="red")
        raise typer.Exit(2)
    try:
        with _client() as c:
            job = c.post(
                "/api/load",
                json={"server": server, "model": model, "force": force, "max_pack": max_pack},
            ).json()
    except httpx.HTTPError as e:
        _bail(e)
    _poll_job(job["id"])


@app.command()
def unload(server: Optional[str] = typer.Option(None, "--server", help="ollama | vllm (default: whatever holds the GPU)")) -> None:
    """Free the GPU."""
    try:
        with _client() as c:
            d = c.post("/api/unload", json={"server": server}).json()
    except httpx.HTTPError as e:
        _bail(e)
    _print_status(d)


@app.command()
def pull(model: str = typer.Argument(..., help="Ollama model to pull")) -> None:
    """Pull/download an Ollama model."""
    try:
        with _client() as c:
            job = c.post("/api/ollama/pull", json={"model": model}).json()
    except httpx.HTTPError as e:
        _bail(e)
    _poll_job(job["id"])


@app.command()
def doctor(local: bool = typer.Option(False, "--local", help="run in-process instead of hitting the API")) -> None:
    """Verify every on-box assumption (run on/against .40)."""
    if local:
        import asyncio

        from .doctor import run_doctor

        rep = asyncio.run(run_doctor()).model_dump()
    else:
        try:
            with _client() as c:
                rep = c.get("/api/doctor").json()
        except httpx.HTTPError as e:
            _bail(e)
    for ch in rep["checks"]:
        glyph, color = {True: ("PASS", "green"), False: ("FAIL", "red"), None: ("WARN", "yellow")}[ch["ok"]]
        typer.secho(f"  [{glyph}] {ch['name']:<24} {ch['detail']}", fg=color)
    typer.echo(f"\n{rep['passed']} passed, {rep['failed']} failed, {rep['warnings']} warnings")
    if rep["failed"]:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print_status(d: dict) -> None:
    owner = d["owner"]
    color = {"free": "green", "ollama": "cyan", "vllm": "magenta", "unknown": "yellow"}.get(owner, "white")
    typer.secho(f"owner: {owner}", fg=color, bold=True)
    g = d["gpu"]
    if g["found"]:
        typer.echo(f"gpu:   {g['used_mb']}/{g['total_mb']} MiB ({g['utilization_pct']}%)")
    else:
        typer.echo(f"gpu:   n/a ({g.get('error', '')})")
    lm = d.get("loaded")
    if lm:
        if lm["server"] == "ollama":
            spill = f", {_gib(lm['on_cpu_bytes'])} on CPU" if lm["spilled"] else " (fully on GPU)"
            typer.echo(f"model: {lm['model']} [ollama] {_gib(lm['on_gpu_bytes'])} on GPU{spill}")
        else:
            typer.echo(f"model: {lm['model']} [vllm]")
    else:
        typer.echo("model: none")
    if d.get("swap_in_progress"):
        typer.secho(f"swap in progress (job {d.get('active_job_id')})", fg="yellow")
    typer.echo(f"ollama_up={d['ollama_up']}  vllm_up={d['vllm_up']}")


def _poll_job(jid: str) -> None:
    seen = 0
    try:
        with _client() as c:
            while True:
                j = c.get(f"/api/jobs/{jid}").json()
                for line in j["log"][seen:]:
                    typer.echo(f"  {line}")
                seen = len(j["log"])
                if j["state"] in ("succeeded", "failed"):
                    if j["state"] == "succeeded":
                        typer.secho(f"OK {j['kind']}", fg="green")
                    else:
                        typer.secho(f"FAILED: {j.get('error', '')}", fg="red")
                        raise typer.Exit(1)
                    return
                time.sleep(1.0)
    except httpx.HTTPError as e:
        _bail(e)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
