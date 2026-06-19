"use strict";

const $ = (id) => document.getElementById(id);
const GIB = (n) => ((n || 0) / 1024 ** 3).toFixed(1) + "G";

// ---- API key (optional, stored locally) ----
const KEY = "llmconfig_api_key";
$("apikey").value = localStorage.getItem(KEY) || "";
$("apikey").addEventListener("change", (e) => localStorage.setItem(KEY, e.target.value.trim()));

function headers(extra) {
  const h = Object.assign({ "Content-Type": "application/json" }, extra || {});
  const k = localStorage.getItem(KEY);
  if (k) h["X-API-Key"] = k;
  return h;
}
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ headers: headers() }, opts));
  if (!r.ok) throw new Error((await r.text()) || r.status);
  return r.json();
}

let busy = false;        // a load/unload/pull job is running (UI does one at a time)
let LANES = [];          // [{id, name, enabled, default}]
const panels = {};       // lane id -> {el, refs, lane}

function log(text, append) {
  const el = $("log");
  el.textContent = append ? el.textContent + "\n" + text : text;
  el.scrollTop = el.scrollHeight;
}

// ---- boot: discover lanes, build a panel each ----
async function boot() {
  try { LANES = await api("/api/lanes"); }
  catch (e) { LANES = [{ id: "primary", name: "GPU", enabled: true, default: null }]; }
  const root = $("lanes");
  root.innerHTML = "";
  LANES.forEach((l) => { const p = buildPanel(l); panels[l.id] = p; root.appendChild(p.el); });
  await refreshAll();
}

function buildPanel(lane) {
  const el = document.createElement("section");
  el.className = "lane";
  el.innerHTML = `
    <div class="lane-head">
      <span class="lane-name">${lane.name || lane.id}</span>
      <span class="badge owner">…</span>
      <div class="vram"><div class="vram-bar"><div class="vram-fill"></div></div><span class="vram-text"></span></div>
      <span class="loaded"></span>
      <button class="btn btn-warn unload" title="Free this GPU">Unload</button>
    </div>
    <div class="lane-cols">
      <div class="col">
        <h3>Ollama <small class="dot ollama-dot"></small></h3>
        <div class="pull">
          <input class="pull-name" placeholder="pull a model, e.g. qwen3:4b" />
          <button class="btn pull-btn">Pull</button>
        </div>
        <div class="list ollama-list"></div>
        <p class="err ollama-err"></p>
      </div>
      <div class="col">
        <h3>vLLM <small class="dot vllm-dot"></small></h3>
        <div class="list vllm-list"></div>
        <p class="err vllm-err"></p>
      </div>
    </div>`;
  const q = (s) => el.querySelector(s);
  const refs = {
    owner: q(".owner"), vramFill: q(".vram-fill"), vramText: q(".vram-text"), loaded: q(".loaded"),
    ollamaDot: q(".ollama-dot"), vllmDot: q(".vllm-dot"),
    ollamaList: q(".ollama-list"), vllmList: q(".vllm-list"),
    ollamaErr: q(".ollama-err"), vllmErr: q(".vllm-err"),
    pullName: q(".pull-name"), pullBtn: q(".pull-btn"), unload: q(".unload"),
  };
  refs.unload.onclick = () => doUnload(lane.id);
  refs.pullBtn.onclick = () => doPull(lane.id, refs.pullName);
  refs.pullName.addEventListener("keydown", (e) => { if (e.key === "Enter") doPull(lane.id, refs.pullName); });
  return { el, refs, lane };
}

function laneDefault(id) {
  const l = LANES.find((x) => x.id === id);
  return l && l.default ? l.default : null;
}

function modelCard(laneId, server, name, meta, status, loaded) {
  const def = laneDefault(laneId);
  const isDefault = def && def.server === server && def.model === name;

  const card = document.createElement("div");
  card.className = "card" + (loaded ? " loaded-card" : "");

  const left = document.createElement("div");
  left.innerHTML = `<div class="name">${name}` +
    (status ? ` <span class="tag ${status}">${status}</span>` : "") +
    `</div><div class="meta">${meta}</div>`;

  const actions = document.createElement("div");
  actions.className = "actions";

  const star = document.createElement("button");
  star.className = "btn star" + (isDefault ? " on" : "");
  star.title = isDefault ? "Startup default (click to clear)" : "Set as startup default";
  star.textContent = isDefault ? "★" : "☆";
  star.disabled = busy;
  star.onclick = () => setDefault(laneId, server, name);

  const btn = document.createElement("button");
  btn.className = "btn";
  btn.textContent = loaded ? "Loaded" : "Load";
  btn.disabled = loaded || busy;
  btn.onclick = () => doLoad(laneId, server, name);

  actions.appendChild(star);
  actions.appendChild(btn);
  card.appendChild(left);
  card.appendChild(actions);
  return card;
}

// ---- status / catalog ----
async function refreshStatus() {
  let d;
  try { d = await api("/api/status"); } catch (e) { return; }
  (d.lanes || []).forEach((l) => {
    const p = panels[l.id];
    if (!p) return;
    const r = p.refs;
    r.owner.textContent = l.owner;
    r.owner.className = "badge owner " + l.owner;
    const g = l.gpu || {};
    r.vramFill.style.width = (g.found ? g.utilization_pct : 0) + "%";
    r.vramText.textContent = g.found ? `${g.used_mb}/${g.total_mb} MiB (${g.utilization_pct}%)` : "GPU n/a";
    const lm = l.loaded;
    if (lm && lm.server === "ollama") {
      const spill = lm.spilled ? `, ${GIB(lm.on_cpu_bytes)} CPU` : " (all GPU)";
      r.loaded.textContent = `${lm.model} · ollama · ${GIB(lm.on_gpu_bytes)} GPU${spill}`;
    } else if (lm) {
      r.loaded.textContent = `${lm.model} · vllm`;
    } else {
      r.loaded.textContent = "no model loaded";
    }
    r.ollamaDot.className = "dot ollama-dot" + (l.ollama_up ? " up" : "");
    r.vllmDot.className = "dot vllm-dot" + (l.vllm_up ? " up" : "");
  });
}

async function refreshModels() {
  for (const lane of LANES) {
    const p = panels[lane.id];
    if (!p) continue;
    const r = p.refs;
    let d;
    try { d = await api("/api/models?lane=" + encodeURIComponent(lane.id)); } catch (e) { continue; }
    r.ollamaList.innerHTML = "";
    (d.ollama || []).forEach((m) => r.ollamaList.appendChild(modelCard(lane.id, "ollama", m.name, GIB(m.size_bytes), "", m.loaded)));
    r.ollamaErr.textContent = d.ollama_error || "";
    r.vllmList.innerHTML = "";
    (d.vllm || []).forEach((a) => r.vllmList.appendChild(modelCard(lane.id, "vllm", a.alias, `→ ${a.served_name}`, a.status, a.loaded)));
    r.vllmErr.textContent = d.vllm_error || "";
  }
}

// ---- actions ----
async function doLoad(laneId, server, model) {
  if (busy) return;
  busy = true; setButtons();
  log(`loading ${model} on ${server} [${laneId}]…`);
  try {
    const job = await api("/api/load", { method: "POST", body: JSON.stringify({ server, model, lane: laneId }) });
    await pollJob(job.id);
  } catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
}

async function doUnload(laneId) {
  if (busy) return;
  busy = true; setButtons(); log(`unloading / freeing GPU [${laneId}]…`);
  try { await api("/api/unload", { method: "POST", body: JSON.stringify({ lane: laneId }) }); log("GPU freed", true); }
  catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
}

async function doPull(laneId, input) {
  const name = input.value.trim();
  if (!name || busy) return;
  busy = true; setButtons(); log(`pulling ${name}…`);
  try {
    const job = await api("/api/ollama/pull", { method: "POST", body: JSON.stringify({ model: name }) });
    await pollJob(job.id);
    input.value = "";
  } catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
}

async function setDefault(laneId, server, model) {
  if (busy) return;
  const cur = laneDefault(laneId);
  const clearing = cur && cur.server === server && cur.model === model;
  try {
    await api(`/api/lanes/${encodeURIComponent(laneId)}/default`, {
      method: "PUT",
      body: JSON.stringify(clearing ? { server: "", model: "" } : { server, model }),
    });
    LANES = await api("/api/lanes");
    log(clearing ? `cleared ${laneId} startup default` : `set ${laneId} startup default → ${model} [${server}]`, true);
    await refreshModels();
  } catch (e) { log("error: " + e.message, true); }
}

async function pollJob(id) {
  let seen = 0;
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    for (let i = seen; i < j.log.length; i++) log(j.log[i], true);
    seen = j.log.length;
    if (j.state === "succeeded" || j.state === "failed") {
      log(j.state === "succeeded" ? "✓ done" : "✗ " + (j.error || "failed"), true);
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function setButtons() {
  document.querySelectorAll(".card .btn").forEach((b) => {
    if (b.textContent !== "Loaded") b.disabled = busy;
  });
  document.querySelectorAll(".lane .unload, .lane .pull-btn").forEach((b) => (b.disabled = busy));
}

// ---- boot ----
async function refreshAll() { await refreshStatus(); await refreshModels(); }
boot();
setInterval(() => { if (!busy) refreshStatus(); }, 2500);
setInterval(() => { if (!busy) refreshModels(); }, 12000);
