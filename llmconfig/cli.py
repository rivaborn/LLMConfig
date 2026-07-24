"""`llmconfig` CLI — a thin client over the REST API (+ `serve` to launch it).

Talks to the app on .40 (default http://127.0.0.1:11430). Point it elsewhere with
--url / $LLMCONFIG_URL (e.g. http://192.168.1.40:11430 over Tailscale).
"""
from __future__ import annotations

import sys
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
def usage(lane: str = typer.Option("primary", "--lane", help="lane mirrored at the top level")) -> None:
    """Per-lane tri-state: free / idle (model loaded, unused) / active (in use)."""
    try:
        with _client() as c:
            d = c.get("/api/usage", params={"lane": lane}).json()
    except httpx.HTTPError as e:
        _bail(e)
    for u in d.get("lanes", []):
        typer.secho(f"{u['lane']}: ", nl=False, bold=True)
        typer.secho(u["state"], fg=_USAGE_COLORS.get(u["state"], "white"), nl=False)
        if u.get("model"):
            typer.echo(f" — {u['model']}{_idle_suffix(u.get('idle_s'))}")
        else:
            typer.echo("")


@app.command()
def models(lane: str = typer.Option("primary", "--lane", help="primary (3090) | companion (3070 Ti)")) -> None:
    """List available Ollama models and vLLM aliases."""
    try:
        with _client() as c:
            d = c.get("/api/models", params={"lane": lane}).json()
    except httpx.HTTPError as e:
        _bail(e)
    typer.secho(f"[lane: {lane}]", fg="bright_black")
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
def gpu(lane: str = typer.Option("primary", "--lane", help="primary (3090) | companion (3070 Ti)")) -> None:
    """Show nvidia-smi state for a lane's GPU."""
    try:
        with _client() as c:
            d = c.get("/api/gpu", params={"lane": lane}).json()
    except httpx.HTTPError as e:
        _bail(e)
    if not d["found"]:
        typer.secho(f"GPU not found: {d.get('error')}", fg="red")
        raise typer.Exit(1)
    util = d.get("utilization_pct")
    util_s = f"  util {util}%" if util is not None else ""
    typer.echo(f"{d['uuid']}\n  {d['used_mb']}/{d['total_mb']} MiB used ({d['vram_pct']}%)  free {d['free_mb']} MiB{util_s}")
    for p in d.get("processes", []):
        typer.echo(f"    pid {p['pid']:>7}  {p['used_mb']:>6} MiB  {p['name']}")


@app.command()
def load(
    server: str = typer.Argument(..., help="ollama | vllm"),
    model: str = typer.Argument(..., help="Ollama tag or vLLM serve.sh alias"),
    lane: str = typer.Option("primary", "--lane", help="primary (3090) | companion (3070 Ti)"),
    force: bool = typer.Option(False, "--force", help="reload even if already active"),
    max_pack: bool = typer.Option(False, "--max-pack", help="fill VRAM (num_gpu) before spilling (Ollama)"),
) -> None:
    """Load a model on a server, evicting everything else from that lane's GPU first."""
    if server not in ("ollama", "vllm"):
        typer.secho("server must be 'ollama' or 'vllm'", fg="red")
        raise typer.Exit(2)
    try:
        with _client() as c:
            job = c.post(
                "/api/load",
                json={"server": server, "model": model, "lane": lane, "force": force, "max_pack": max_pack},
            ).json()
    except httpx.HTTPError as e:
        _bail(e)
    _poll_job(job["id"])


@app.command()
def unload(
    server: Optional[str] = typer.Option(None, "--server", help="ollama | vllm (default: whatever holds the GPU)"),
    lane: str = typer.Option("primary", "--lane", help="primary (3090) | companion (3070 Ti)"),
) -> None:
    """Free a lane's GPU."""
    try:
        with _client() as c:
            d = c.post("/api/unload", json={"server": server, "lane": lane}).json()
    except httpx.HTTPError as e:
        _bail(e)
    _print_status(d)


@app.command(name="companion-default")
def companion_default(
    server: Optional[str] = typer.Argument(None, help="ollama | vllm (omit to show current)"),
    model: Optional[str] = typer.Argument(None, help="Ollama tag or vLLM alias (empty clears)"),
    lane: str = typer.Option("companion", "--lane", help="lane to configure"),
) -> None:
    """Show or set a lane's default model (auto-loads on startup)."""
    try:
        with _client() as c:
            if server is None:
                d = c.get(f"/api/lanes/{lane}/default").json()
            else:
                d = c.put(f"/api/lanes/{lane}/default", json={"server": server, "model": model or ""}).json()
    except httpx.HTTPError as e:
        _bail(e)
    dflt = d.get("default")
    if dflt:
        typer.echo(f"{lane} default: {dflt['model']} [{dflt['server']}]")
    else:
        typer.echo(f"{lane} default: none")


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
# leases — claim a unit so other callers know not to take it
# --------------------------------------------------------------------------- #
lease_app = typer.Typer(add_completion=False, help="Claim / inspect / release unit leases.")
app.add_typer(lease_app, name="lease")


def _lease_err(r: httpx.Response) -> None:
    """Print a lease API error and exit 1. Bodies are {"detail": {...}}."""
    try:
        d = r.json().get("detail", {})
    except ValueError:
        d = {}
    if isinstance(d, dict):
        typer.secho(f"{d.get('error', r.status_code)}: {d.get('message', r.text)}", fg="red")
        held = d.get("lease")
        if held:
            typer.secho(f"  held by '{held.get('holder')}' "
                        f"({'non-preemptible' if not held.get('preemptible', True) else 'preemptible'}"
                        f", priority {held.get('priority', 0)})", fg="yellow")
    else:
        typer.secho(f"{r.status_code}: {d or r.text}", fg="red")
    raise typer.Exit(1)


@lease_app.command("claim")
def lease_claim(
    holder: str = typer.Option(..., "--holder", help="who you are, e.g. nightly-eval"),
    unit: str = typer.Option("primary", "--unit", help="unit id (primary | companion | spark1 …)"),
    minutes: float = typer.Option(10.0, "--minutes", help="renewable leash (ttl)"),
    expect_minutes: float = typer.Option(0.0, "--expect-minutes",
                                         help="how long you'll really need it (hint only)"),
    preempt: bool = typer.Option(True, "--preempt/--no-preempt",
                                 help="--no-preempt = must not be interrupted"),
    priority: int = typer.Option(0, "--priority", help="0..100; higher wins among preemptible"),
    server: Optional[str] = typer.Option(None, "--server"),
    model: str = typer.Option("", "--model"),
    note: str = typer.Option("", "--note"),
    force: bool = typer.Option(False, "--force", help="steal an equal-priority preemptible lease"),
    free_on_preempt: bool = typer.Option(False, "--free-on-preempt",
                                         help="also evict whatever is loaded, once granted"),
    quiet: bool = typer.Option(False, "--quiet", help="print only the lease id"),
) -> None:
    """Claim a unit. Prints the lease id — send it as X-LLM-Lease on /v1 requests."""
    body = {
        "unit": unit, "holder": holder, "preemptible": preempt, "priority": priority,
        "ttl_s": minutes * 60.0, "expected_duration_s": expect_minutes * 60.0,
        "model": model, "note": note, "force": force, "free_on_preempt": free_on_preempt,
    }
    if server:
        body["server"] = server
    try:
        with _client() as c:
            r = c.post("/api/leases", json=body)
    except httpx.HTTPError as e:
        _bail(e)
    if r.status_code >= 400:
        _lease_err(r)
    d = r.json()
    lease = d["lease"]
    if quiet:
        typer.echo(lease["id"])
        return
    typer.secho(f"lease {lease['id']} on {lease['unit']} for '{lease['holder']}'", fg="green", bold=True)
    typer.echo(f"  {'non-preemptible' if not lease['preemptible'] else 'preemptible'}"
               f", priority {lease['priority']}, ttl {lease['ttl_s'] / 60:.0f}m")
    if d.get("displaced"):
        typer.secho(f"  displaced '{d['displaced']['holder']}' (lease {d['displaced']['id']})", fg="yellow")
    if d.get("busy_with"):
        # A non-preemptible lease is a FORWARD guarantee: work already running keeps
        # the unit until it finishes.
        typer.secho("  note: the unit is already busy — work in flight finishes first "
                    f"({d['busy_with'].get('usage')})", fg="yellow")


@lease_app.command("list")
def lease_list(
    unit: Optional[str] = typer.Option(None, "--unit"),
    active: bool = typer.Option(False, "--active", help="only live leases"),
) -> None:
    # Only send `unit` when set: httpx serializes None as `unit=` (empty string),
    # which the server would read as a unit literally named "".
    params: dict = {"active": active}
    if unit:
        params["unit"] = unit
    try:
        with _client() as c:
            d = c.get("/api/leases", params=params).json()
    except httpx.HTTPError as e:
        _bail(e)
    leases = d.get("leases", [])
    if not leases:
        typer.secho("no leases", fg="bright_black")
        return
    for l in leases:
        _print_lease(l)


@lease_app.command("show")
def lease_show(lease_id: str) -> None:
    try:
        with _client() as c:
            r = c.get(f"/api/leases/{lease_id}")
    except httpx.HTTPError as e:
        _bail(e)
    if r.status_code == 404:
        _lease_err(r)
    _print_lease(r.json(), verbose=True)


@lease_app.command("renew")
def lease_renew(lease_id: str, minutes: Optional[float] = typer.Option(None, "--minutes")) -> None:
    body = {} if minutes is None else {"ttl_s": minutes * 60.0}
    try:
        with _client() as c:
            r = c.post(f"/api/leases/{lease_id}/renew", json=body)
    except httpx.HTTPError as e:
        _bail(e)
    if r.status_code >= 400:
        _lease_err(r)
    typer.secho(f"renewed {lease_id} (+{r.json()['ttl_s'] / 60:.0f}m)", fg="green")


@lease_app.command("release")
def lease_release(lease_id: str) -> None:
    try:
        with _client() as c:
            r = c.delete(f"/api/leases/{lease_id}")
    except httpx.HTTPError as e:
        _bail(e)
    if r.status_code >= 400:
        _lease_err(r)
    typer.secho(f"released {lease_id}", fg="green")


@lease_app.command("revoke")
def lease_revoke(
    lease_id: str,
    reason: str = typer.Option("admin", "--reason"),
    by: str = typer.Option("operator", "--by"),
    free: bool = typer.Option(False, "--free", help="also evict whatever is loaded"),
) -> None:
    """Operator break-glass — takes a unit back even from a non-preemptible lease."""
    try:
        with _client() as c:
            r = c.post(f"/api/leases/{lease_id}/revoke",
                       json={"reason": reason, "by": by, "free": free})
    except httpx.HTTPError as e:
        _bail(e)
    if r.status_code >= 400:
        _lease_err(r)
    typer.secho(f"revoked {lease_id}", fg="yellow")


@lease_app.command("wait")
def lease_wait(lease_id: str) -> None:
    """Block until the lease ends, then exit with a code saying HOW it ended.

    This is the scriptable half of "was my job kicked off a unit?":
      0 released · 3 revoked · 4 expired · 2 unreachable/unknown
    """
    try:
        with _client() as c:
            while True:
                r = c.get(f"/api/leases/{lease_id}")
                if r.status_code == 404:
                    typer.secho(f"lease {lease_id} is unknown (server restarted?)", fg="red")
                    raise typer.Exit(2)
                l = r.json()
                if l["state"] != "active":
                    _print_lease(l, verbose=True)
                    raise typer.Exit({"released": 0, "revoked": 3, "expired": 4}.get(l["state"], 1))
                time.sleep(2.0)
    except httpx.HTTPError as e:
        _bail(e)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_USAGE_COLORS = {"free": "green", "idle": "yellow", "active": "cyan"}
_LEASE_COLORS = {"active": "cyan", "released": "green", "revoked": "red", "expired": "yellow"}


def _print_lease(l: dict, verbose: bool = False) -> None:
    color = _LEASE_COLORS.get(l["state"], "white")
    kind = "non-preemptible" if not l.get("preemptible", True) else "preemptible"
    typer.secho(f"[{l['unit']}] {l['id']}  {l['state']}  '{l['holder']}' ({kind})",
                fg=color, bold=True)
    if l.get("revoked_by") or l.get("revoked_reason"):
        typer.secho(f"  taken by '{l.get('revoked_by') or '?'}' — {l.get('revoked_reason') or ''}",
                    fg="red")
    if not verbose:
        return
    if l.get("model"):
        typer.echo(f"  model: {l['model']} [{l.get('server') or '?'}]")
    typer.echo(f"  ttl {l['ttl_s'] / 60:.0f}m, renewed {l.get('renew_count', 0)}×, "
               f"{l.get('requests', 0)} request(s)")
    if l.get("expected_duration_s"):
        typer.echo(f"  expected duration: {l['expected_duration_s'] / 60:.0f}m")
    if l.get("note"):
        typer.secho(f"  note: {l['note']}", fg="bright_black")


def _idle_suffix(idle_s: object) -> str:
    if not isinstance(idle_s, (int, float)):
        return ""
    return f" (idle {idle_s:.0f}s)" if idle_s < 120 else f" (idle {idle_s / 60:.1f}m)"


def _print_lane(l: dict) -> None:
    owner = l["owner"]
    color = {"free": "green", "ollama": "cyan", "vllm": "magenta", "unknown": "yellow"}.get(owner, "white")
    label = f"{l.get('id', 'primary')}" + (f" / {l['name']}" if l.get("name") else "")
    use = l.get("usage")
    suffix = ""
    if use:
        suffix = f"  [{use}{_idle_suffix(l.get('idle_s')) if use == 'idle' else ''}]"
    typer.secho(f"[{label}] owner: {owner}{suffix}", fg=color, bold=True)
    g = l["gpu"]
    if g["found"]:
        typer.echo(f"  gpu:   {g['used_mb']}/{g['total_mb']} MiB ({g['vram_pct']}%)")
    else:
        typer.echo(f"  gpu:   n/a ({g.get('error', '')})")
    lm = l.get("loaded")
    if lm:
        if lm["server"] == "ollama":
            spill = f", {_gib(lm['on_cpu_bytes'])} on CPU" if lm["spilled"] else " (fully on GPU)"
            typer.echo(f"  model: {lm['model']} [ollama] {_gib(lm['on_gpu_bytes'])} on GPU{spill}")
        else:
            typer.echo(f"  model: {lm['model']} [vllm]")
    else:
        typer.echo("  model: none")
    if l.get("swap_in_progress"):
        typer.secho(f"  swap in progress (job {l.get('active_job_id')})", fg="yellow")
    held = l.get("lease")
    if held:
        kind = "non-preemptible" if not held.get("preemptible", True) else "preemptible"
        typer.secho(f"  lease: '{held['holder']}' ({kind}, {held.get('expires_in_s', 0) / 60:.0f}m left)"
                    f" [{held['id']}]", fg="magenta")
    typer.echo(f"  ollama_up={l['ollama_up']}  vllm_up={l['vllm_up']}")


def _print_status(d: dict) -> None:
    lanes = d.get("lanes")
    if lanes:
        for l in lanes:
            _print_lane(l)
    else:  # legacy single-lane response shape
        _print_lane({**d, "id": "primary"})


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
    # On Windows the console defaults to a legacy code page (cp1252), which turns
    # the report glyphs (— … → ●) into mojibake. Force UTF-8 on our streams.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    app()


if __name__ == "__main__":
    main()
