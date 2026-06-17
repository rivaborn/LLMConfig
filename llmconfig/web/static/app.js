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

let busy = false; // a load/unload/pull job is running

function log(text, append) {
  const el = $("log");
  el.textContent = append ? el.textContent + "\n" + text : text;
  el.scrollTop = el.scrollHeight;
}

// ---- status ----
async function refreshStatus() {
  let d;
  try { d = await api("/api/status"); } catch (e) { return; }
  const owner = d.owner;
  const ob = $("owner");
  ob.textContent = owner;
  ob.className = "badge " + owner;

  const g = d.gpu || {};
  $("vram-fill").style.width = (g.found ? g.utilization_pct : 0) + "%";
  $("vram-text").textContent = g.found ? `${g.used_mb}/${g.total_mb} MiB (${g.utilization_pct}%)` : "GPU n/a";

  const lm = d.loaded;
  if (lm && lm.server === "ollama") {
    const spill = lm.spilled ? `, ${GIB(lm.on_cpu_bytes)} CPU` : " (all GPU)";
    $("loaded").textContent = `${lm.model} · ollama · ${GIB(lm.on_gpu_bytes)} GPU${spill}`;
  } else if (lm) {
    $("loaded").textContent = `${lm.model} · vllm`;
  } else {
    $("loaded").textContent = "no model loaded";
  }
  $("ollama-up").className = "dot" + (d.ollama_up ? " up" : "");
  $("vllm-up").className = "dot" + (d.vllm_up ? " up" : "");
}

// ---- catalog ----
async function refreshModels() {
  let d;
  try { d = await api("/api/models"); } catch (e) { return; }
  const ol = $("ollama-list"); ol.innerHTML = "";
  (d.ollama || []).forEach((m) => ol.appendChild(modelCard("ollama", m.name, GIB(m.size_bytes), "", m.loaded)));
  $("ollama-err").textContent = d.ollama_error || "";

  const vl = $("vllm-list"); vl.innerHTML = "";
  (d.vllm || []).forEach((a) => vl.appendChild(
    modelCard("vllm", a.alias, `→ ${a.served_name}`, a.status, a.loaded)
  ));
  $("vllm-err").textContent = d.vllm_error || "";
}

function modelCard(server, name, meta, status, loaded) {
  const card = document.createElement("div");
  card.className = "card" + (loaded ? " loaded-card" : "");
  const left = document.createElement("div");
  left.innerHTML = `<div class="name">${name}` +
    (status ? ` <span class="tag ${status}">${status}</span>` : "") +
    `</div><div class="meta">${meta}</div>`;
  const btn = document.createElement("button");
  btn.className = "btn";
  btn.textContent = loaded ? "Loaded" : "Load";
  btn.disabled = loaded || busy;
  btn.onclick = () => doLoad(server, name);
  card.appendChild(left); card.appendChild(btn);
  return card;
}

// ---- actions ----
async function doLoad(server, model) {
  if (busy) return;
  busy = true; setButtons();
  log(`loading ${model} on ${server}…`);
  try {
    const job = await api("/api/load", { method: "POST", body: JSON.stringify({ server, model }) });
    await pollJob(job.id);
  } catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
}

async function doUnload() {
  if (busy) return;
  busy = true; setButtons(); log("unloading / freeing GPU…");
  try { await api("/api/unload", { method: "POST", body: JSON.stringify({}) }); log("GPU freed", true); }
  catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
}

async function doPull() {
  const name = $("pull-name").value.trim();
  if (!name || busy) return;
  busy = true; setButtons(); log(`pulling ${name}…`);
  try {
    const job = await api("/api/ollama/pull", { method: "POST", body: JSON.stringify({ model: name }) });
    await pollJob(job.id);
    $("pull-name").value = "";
  } catch (e) { log("error: " + e.message, true); }
  busy = false; await refreshAll(); setButtons();
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
  $("unload").disabled = busy;
  $("pull-btn").disabled = busy;
}

// ---- boot ----
$("unload").onclick = doUnload;
$("pull-btn").onclick = doPull;
$("pull-name").addEventListener("keydown", (e) => { if (e.key === "Enter") doPull(); });

async function refreshAll() { await Promise.all([refreshStatus(), refreshModels()]); }
refreshAll();
setInterval(() => { if (!busy) refreshStatus(); }, 2500);
setInterval(() => { if (!busy) refreshModels(); }, 12000);
