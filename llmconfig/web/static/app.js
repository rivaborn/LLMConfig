"use strict";

// LLMConfig UI shell.
//
// Layout is driven entirely by /api/lanes, which enumerates every LLM unit —
// the local GPU lanes (RTX 3090 / 3070 Ti) and each remote DGX Spark. From that
// list this builds: a Home dashboard (one box per unit with a quick-switch
// dropdown), one control tab per unit, and the Monitor tab.
//
// This file owns tab switching and calls into monitor.js (loaded after it) via
// window.MonitorHooks, so the Monitor only polls while its tab is visible.

const $ = (id) => document.getElementById(id);
const GIB = (n) => ((n || 0) / 1024 ** 3).toFixed(1) + "G";
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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

// ---- state ----
let UNITS = [];                  // [{id, name, kind, host, enabled, default}]
const panels = {};               // unit id -> {el, refs, unit}
const cards = {};                // unit id -> {el, refs, unit}   (Home dashboard)
const busyUnits = new Set();     // units with a job in flight (per-unit, not global)
let currentView = "home";

const isBusy = (id) => busyUnits.has(id);
const anyBusy = () => busyUnits.size > 0;

function log(text, append) {
  const el = $("log");
  el.textContent = append ? el.textContent + "\n" + text : text;
  el.scrollTop = el.scrollHeight;
}

// ---- tabs / views --------------------------------------------------------
function showView(name) {
  currentView = name;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".view").forEach((v) => {
    const on = v.id === "view-" + name;
    v.classList.toggle("active", on);
    v.hidden = !on;
  });
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  // monitor.js defines these; it loads after this file, and boot() is async, so
  // the hooks are always registered by the time a tab can be clicked.
  const hooks = window.MonitorHooks;
  if (hooks) (name === "monitor" ? hooks.start : hooks.stop)();
}

function buildTabs() {
  const nav = $("tabs");
  nav.innerHTML = "";
  const mk = (view, label, title) => {
    const b = document.createElement("button");
    b.className = "tab";
    b.dataset.view = view;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", "false");
    b.textContent = label;
    if (title) b.title = title;
    b.addEventListener("click", () => showView(view));
    nav.appendChild(b);
    return b;
  };
  mk("home", "Home", "All units at a glance");
  UNITS.forEach((u) =>
    mk(u.id, u.name || u.id, u.kind === "spark" ? `DGX Spark · ${u.host}` : "Local GPU lane"));
  mk("monitor", "Monitor", "Telemetry for every unit");
}

// ---- Home dashboard ------------------------------------------------------
function buildCard(unit) {
  const el = document.createElement("article");
  el.className = "unit-card";
  el.innerHTML = `
    <div class="uc-head">
      <span class="uc-name">${esc(unit.name || unit.id)}</span>
      <span class="uc-kind ${unit.kind === "spark" ? "spark" : "gpu"}">${unit.kind === "spark" ? "SPARK" : "GPU"}</span>
      <span class="badge owner">…</span>
    </div>
    <div class="uc-sub">${unit.kind === "spark" ? esc(unit.host) : "local"}</div>
    <div class="uc-loaded">…</div>
    <div class="vram"><div class="vram-bar"><div class="vram-fill"></div></div><span class="vram-text"></span></div>
    <div class="uc-switch">
      <select class="uc-select" aria-label="Model for ${esc(unit.name || unit.id)}"></select>
      <button class="btn uc-load" title="Load the selected model">Switch</button>
      <button class="btn btn-warn uc-unload" title="Free this unit">Unload</button>
    </div>`;
  const q = (s) => el.querySelector(s);
  const refs = {
    owner: q(".owner"), loaded: q(".uc-loaded"),
    vramFill: q(".vram-fill"), vramText: q(".vram-text"),
    select: q(".uc-select"), load: q(".uc-load"), unload: q(".uc-unload"),
  };
  refs.load.onclick = () => {
    const v = refs.select.value;
    if (!v) return;
    const [server, ...rest] = v.split("::");
    doLoad(unit.id, server, rest.join("::"));
  };
  refs.unload.onclick = () => doUnload(unit.id);
  // Jump to the unit's own tab by clicking its name.
  q(".uc-name").addEventListener("click", () => showView(unit.id));
  return { el, refs, unit };
}

// ---- per-unit control panels --------------------------------------------
function buildPanel(unit) {
  const el = document.createElement("section");
  el.className = "view";
  el.id = "view-" + unit.id;
  el.setAttribute("role", "tabpanel");
  el.hidden = true;
  el.innerHTML = unit.kind === "spark" ? sparkPanelHtml(unit) : gpuPanelHtml(unit);

  const q = (s) => el.querySelector(s);
  const refs = {
    owner: q(".owner"), vramFill: q(".vram-fill"), vramText: q(".vram-text"), loaded: q(".loaded"),
    unload: q(".unload"),
    ollamaDot: q(".ollama-dot"), vllmDot: q(".vllm-dot"),
    ollamaList: q(".ollama-list"), vllmList: q(".vllm-list"),
    ollamaErr: q(".ollama-err"), vllmErr: q(".vllm-err"),
    pullName: q(".pull-name"), pullBtn: q(".pull-btn"),
    sparkList: q(".spark-list"), sparkErr: q(".spark-err"),
  };
  refs.unload.onclick = () => doUnload(unit.id);
  if (refs.pullBtn) {
    refs.pullBtn.onclick = () => doPull(unit.id, refs.pullName);
    refs.pullName.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doPull(unit.id, refs.pullName);
    });
  }
  return { el, refs, unit };
}

function headHtml(unit, hint) {
  return `
    <p class="hint">${hint}</p>
    <section class="lane">
      <div class="lane-head">
        <span class="lane-name">${esc(unit.name || unit.id)}</span>
        <span class="badge owner">…</span>
        <div class="vram"><div class="vram-bar"><div class="vram-fill"></div></div><span class="vram-text"></span></div>
        <span class="loaded"></span>
        <button class="btn btn-warn unload" title="Free this unit">Unload</button>
      </div>`;
}

function gpuPanelHtml(unit) {
  return headHtml(unit, "This GPU runs one model at a time (Ollama ⇄ vLLM); lanes are independent.") + `
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
      </div>
    </section>`;
}

function sparkPanelHtml(unit) {
  return headHtml(unit,
    `DGX Spark node at ${esc(unit.host)} — driven by sparkrun. The node serves one workload at a time.`) + `
      <div class="lane-cols one">
        <div class="col">
          <h3>Curated recipes <small class="muted-inline">sparkrun</small></h3>
          <div class="list spark-list"></div>
          <p class="err spark-err"></p>
        </div>
      </div>
    </section>`;
}

function unitDefault(id) {
  const u = UNITS.find((x) => x.id === id);
  return u && u.default ? u.default : null;
}

function modelCard(unitId, server, name, meta, status, loaded) {
  const def = unitDefault(unitId);
  const isDefault = def && def.server === server && def.model === name;

  const card = document.createElement("div");
  card.className = "card" + (loaded ? " loaded-card" : "");

  const left = document.createElement("div");
  left.innerHTML = `<div class="name">${esc(name)}` +
    (status ? ` <span class="tag ${esc(status)}">${esc(status)}</span>` : "") +
    `</div><div class="meta">${esc(meta)}</div>`;

  const actions = document.createElement("div");
  actions.className = "actions";

  const star = document.createElement("button");
  star.className = "btn star" + (isDefault ? " on" : "");
  star.title = isDefault ? "Startup default (click to clear)" : "Set as startup default";
  star.textContent = isDefault ? "★" : "☆";
  star.disabled = isBusy(unitId);
  star.onclick = () => setDefault(unitId, server, name);

  const btn = document.createElement("button");
  btn.className = "btn";
  btn.textContent = loaded ? "Loaded" : "Load";
  btn.disabled = loaded || isBusy(unitId);
  btn.onclick = () => doLoad(unitId, server, name);

  actions.appendChild(star);
  actions.appendChild(btn);
  card.appendChild(left);
  card.appendChild(actions);
  return card;
}

// ---- status --------------------------------------------------------------
function describeLoaded(lm) {
  if (!lm) return "no model loaded";
  // Served names are chosen per unit and CAN collide (the 3090 and a Spark both
  // served "gemma-4-26b" from different weights), so show the backend's root
  // whenever it differs — that's the only thing that identifies the real build.
  const root = lm.root && lm.root !== lm.model ? `  ←  ${lm.root}` : "";
  if (lm.server === "ollama") {
    const spill = lm.spilled ? `, ${GIB(lm.on_cpu_bytes)} CPU` : " (all GPU)";
    return `${lm.model} · ollama · ${GIB(lm.on_gpu_bytes)} GPU${spill}${root}`;
  }
  return `${lm.model} · ${lm.server}${root}`;
}

async function refreshStatus() {
  let d;
  try { d = await api("/api/status"); } catch (e) { return; }
  (d.lanes || []).forEach((l) => {
    const g = l.gpu || {};
    const offline = l.kind === "spark" && l.reachable === false;
    const ownerText = offline ? "offline" : l.owner;
    const vramText = g.found ? `${g.used_mb}/${g.total_mb} MiB (${g.vram_pct}%)`
                             : (offline ? "unreachable" : "GPU n/a");

    const c = cards[l.id];
    if (c) {
      c.el.classList.toggle("offline", offline);
      c.refs.owner.textContent = ownerText;
      c.refs.owner.className = "badge owner " + (offline ? "unknown" : l.owner);
      c.refs.loaded.textContent = l.loaded ? describeLoaded(l.loaded) : "None";
      c.refs.loaded.classList.toggle("none", !l.loaded);
      c.refs.vramFill.style.width = (g.found ? g.vram_pct : 0) + "%";
      c.refs.vramText.textContent = vramText;
      c.refs.unload.disabled = isBusy(l.id) || !l.loaded;
      c.refs.load.disabled = isBusy(l.id);
    }

    const p = panels[l.id];
    if (p) {
      const r = p.refs;
      r.owner.textContent = ownerText;
      r.owner.className = "badge owner " + (offline ? "unknown" : l.owner);
      r.vramFill.style.width = (g.found ? g.vram_pct : 0) + "%";
      r.vramText.textContent = vramText;
      r.loaded.textContent = describeLoaded(l.loaded);
      if (r.ollamaDot) r.ollamaDot.className = "dot ollama-dot" + (l.ollama_up ? " up" : "");
      if (r.vllmDot) r.vllmDot.className = "dot vllm-dot" + (l.vllm_up ? " up" : "");
      r.unload.disabled = isBusy(l.id);
    }
  });
}

// ---- catalog -------------------------------------------------------------
function fillSelect(sel, options, loadedValue) {
  // Preserve the user's pick across polls; otherwise follow what's loaded.
  const prev = sel.value;
  sel.innerHTML = "";
  if (!options.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "no models available";
    sel.appendChild(o);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  options.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.textContent = o.label;
    sel.appendChild(opt);
  });
  const want = options.some((o) => o.value === prev) ? prev : (loadedValue || options[0].value);
  sel.value = want;
}

async function refreshModels() {
  for (const unit of UNITS) {
    let d;
    try { d = await api("/api/models?lane=" + encodeURIComponent(unit.id)); } catch (e) { continue; }

    const opts = [];
    let loadedValue = "";
    (d.ollama || []).forEach((m) => {
      opts.push({ value: `ollama::${m.name}`, label: `${m.name}  ·  ollama` });
      if (m.loaded) loadedValue = `ollama::${m.name}`;
    });
    (d.vllm || []).forEach((a) => {
      opts.push({ value: `vllm::${a.alias}`, label: `${a.alias}  ·  vllm` });
      if (a.loaded) loadedValue = `vllm::${a.alias}`;
    });
    (d.spark || []).forEach((m) => {
      opts.push({ value: `spark::${m.alias}`, label: `${m.alias}  ·  spark` });
      if (m.loaded) loadedValue = `spark::${m.alias}`;
    });

    const c = cards[unit.id];
    if (c) fillSelect(c.refs.select, opts, loadedValue);

    const p = panels[unit.id];
    if (!p) continue;
    const r = p.refs;
    if (r.sparkList) {
      r.sparkList.innerHTML = "";
      (d.spark || []).forEach((m) =>
        r.sparkList.appendChild(
          modelCard(unit.id, "spark", m.alias, `→ ${m.served_name}${m.tp > 1 ? ` · tp${m.tp}` : ""}`,
                    m.status, m.loaded)));
      r.sparkErr.textContent = d.spark_error || "";
      continue;
    }
    r.ollamaList.innerHTML = "";
    (d.ollama || []).forEach((m) =>
      r.ollamaList.appendChild(modelCard(unit.id, "ollama", m.name, GIB(m.size_bytes), "", m.loaded)));
    r.ollamaErr.textContent = d.ollama_error || "";
    r.vllmList.innerHTML = "";
    (d.vllm || []).forEach((a) =>
      r.vllmList.appendChild(modelCard(unit.id, "vllm", a.alias, `→ ${a.served_name}`, a.status, a.loaded)));
    r.vllmErr.textContent = d.vllm_error || "";
  }
}

// ---- actions -------------------------------------------------------------
function markBusy(unitId, on) {
  if (on) busyUnits.add(unitId); else busyUnits.delete(unitId);
  setButtons();
}

async function doLoad(unitId, server, model) {
  if (isBusy(unitId)) return;
  markBusy(unitId, true);
  log(`loading ${model} on ${server} [${unitId}]…`);
  try {
    const job = await api("/api/load", { method: "POST", body: JSON.stringify({ server, model, lane: unitId }) });
    await pollJob(job.id);
  } catch (e) { log("error: " + e.message, true); }
  markBusy(unitId, false);
  await refreshAll();
}

async function doUnload(unitId) {
  if (isBusy(unitId)) return;
  markBusy(unitId, true);
  log(`unloading / freeing [${unitId}]…`);
  try {
    await api("/api/unload", { method: "POST", body: JSON.stringify({ lane: unitId }) });
    log("unit freed", true);
  } catch (e) { log("error: " + e.message, true); }
  markBusy(unitId, false);
  await refreshAll();
}

async function doPull(unitId, input) {
  const name = input.value.trim();
  if (!name || isBusy(unitId)) return;
  markBusy(unitId, true);
  log(`pulling ${name}…`);
  try {
    const job = await api("/api/ollama/pull", { method: "POST", body: JSON.stringify({ model: name }) });
    await pollJob(job.id);
    input.value = "";
  } catch (e) { log("error: " + e.message, true); }
  markBusy(unitId, false);
  await refreshAll();
}

async function setDefault(unitId, server, model) {
  if (isBusy(unitId)) return;
  const cur = unitDefault(unitId);
  const clearing = cur && cur.server === server && cur.model === model;
  try {
    await api(`/api/lanes/${encodeURIComponent(unitId)}/default`, {
      method: "PUT",
      body: JSON.stringify(clearing ? { server: "", model: "" } : { server, model }),
    });
    UNITS = await api("/api/lanes");
    log(clearing ? `cleared ${unitId} startup default` : `set ${unitId} startup default → ${model} [${server}]`, true);
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
  // Disable only the units that are actually busy — a load on the 3090 must not
  // freeze the controls for a Spark.
  Object.entries(panels).forEach(([id, p]) => {
    const b = isBusy(id);
    p.el.querySelectorAll(".card .btn").forEach((x) => {
      if (x.textContent !== "Loaded") x.disabled = b;
    });
    p.el.querySelectorAll(".unload, .pull-btn").forEach((x) => (x.disabled = b));
  });
  Object.entries(cards).forEach(([id, c]) => {
    const b = isBusy(id);
    c.refs.load.disabled = b;
    c.refs.unload.disabled = b;
    c.el.classList.toggle("busy", b);
  });
}

// ---- boot ----------------------------------------------------------------
async function refreshAll() { await refreshStatus(); await refreshModels(); }

async function boot() {
  try { UNITS = await api("/api/lanes"); }
  catch (e) { UNITS = [{ id: "primary", name: "GPU", kind: "gpu", host: "", enabled: true, default: null }]; }

  buildTabs();

  const grid = $("unit-grid");
  const views = $("unit-views");
  grid.innerHTML = "";
  views.innerHTML = "";
  UNITS.forEach((u) => {
    const c = buildCard(u); cards[u.id] = c; grid.appendChild(c.el);
    const p = buildPanel(u); panels[u.id] = p; views.appendChild(p.el);
  });

  // Deep-link support: #<unitId> / #monitor / #home. Unknown hashes fall back Home.
  const want = (location.hash || "#home").slice(1);
  const known = ["home", "monitor", ...UNITS.map((u) => u.id)];
  showView(known.includes(want) ? want : "home");

  await refreshAll();
}

boot();
// Status is read-only and disabled-state is computed per unit, so keep polling even
// while a unit is busy — a 15-minute Spark load must not freeze the other five
// units' cards. The model catalog (which rebuilds DOM lists) still waits.
setInterval(() => refreshStatus(), 2500);
setInterval(() => { if (!anyBusy()) refreshModels(); }, 12000);
