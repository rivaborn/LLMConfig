"use strict";

// Monitor tab — GPU/LLM telemetry console. Reuses api()/headers() from app.js
// (shared global scope) and only polls while its tab is visible.

const mq = (s) => document.querySelector(s);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const MON = {
  active: false,
  metric: "temp",      // temp | power | vram | llm
  window: 14400,       // seconds
  snap: null,
  hist: null,
  timer: null,
};

// Core boost-clock throttle point for the RTX 3090 — a reference line, not a limit.
const TEMP_REDLINE = 83;

// ---- tab switching -------------------------------------------------------
function showView(name) {
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
  MON.active = name === "monitor";
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (MON.active) startMonitor();
  else stopMonitor();
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => showView(t.dataset.view))
);
// Deep-link the active tab so the Monitor view is bookmarkable / shareable.
if (location.hash === "#monitor") showView("monitor");

// ---- segmented controls --------------------------------------------------
function wireSeg(id, key, parse) {
  mq(id).addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn");
    if (!b) return;
    mq(id).querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
    MON[key] = parse(b.dataset[key]);
    if (key === "window") loadHistory();
    renderCharts();
  });
}
wireSeg("#metric-seg", "metric", (v) => v);
wireSeg("#window-seg", "window", (v) => parseFloat(v));

// ---- polling -------------------------------------------------------------
function startMonitor() {
  if (MON.timer) return;
  const tick = async () => { await loadSnapshot(); };
  tick();
  loadHistory();
  let n = 0;
  MON.timer = setInterval(async () => {
    if (!MON.active) return;
    await loadSnapshot();
    if (++n % 2 === 0) await loadHistory(); // history half as often as readouts
  }, 3000);
}
function stopMonitor() {
  if (MON.timer) { clearInterval(MON.timer); MON.timer = null; }
}

async function loadSnapshot() {
  try { MON.snap = await api("/api/monitor"); } catch (e) { return; }
  renderInstruments();
  renderBanner();
  renderNote();
}
async function loadHistory() {
  try { MON.hist = await api("/api/monitor/history?window=" + MON.window); } catch (e) { return; }
  renderCharts();
}

// ---- formatting helpers --------------------------------------------------
const fGiB = (mb) => (mb / 1024).toFixed(1);
const num = (v, d = 0) => (v == null ? "—" : v.toFixed(d));

function tempClass(t) {
  if (t == null) return "";
  if (t >= 90) return "crit";
  if (t >= TEMP_REDLINE) return "hot";
  return "";
}

// ---- instruments (live readouts) ----------------------------------------
function renderInstruments() {
  const root = mq("#instruments");
  if (!MON.snap || !MON.snap.gpus.length) {
    root.innerHTML = `<p class="chart-empty">No GPU telemetry yet — the sampler warms up a few seconds after start.</p>`;
    return;
  }
  const cHot = cssVar("--trace-hotspot"), cJun = cssVar("--trace-junction"), cPow = cssVar("--trace-power");
  root.innerHTML = MON.snap.gpus.map((g) => {
    const sub = (label, val, sw) =>
      `<div class="ro"><span class="ro-label">${label}</span><span class="ro-val">${
        sw ? `<i class="swatch" style="background:${sw}"></i>` : ""
      }${val}</span></div>`;
    const agg = g.temp_max_24h != null
      ? `<span class="dim">max ${num(g.temp_max_24h)}° · avg1h ${num(g.temp_avg_1h)}°</span>` : "";
    return `
      <div class="inst">
        <div class="inst-head">
          <span class="inst-name">${g.name}</span>
          <span class="inst-idx">#${g.index}</span>
        </div>
        <div class="inst-temp ${tempClass(g.temp_c)}">${num(g.temp_c)}<span class="unit">°C</span></div>
        <div class="readouts">
          ${sub("hotspot", num(g.hotspot_c) + "°", cHot)}
          ${sub("junction", num(g.junction_c) + "°", cJun)}
          ${sub("power", num(g.power_w) + " W", cPow)}
          ${sub("util", num(g.util_pct) + " %")}
        </div>
        <div class="inst-vram">
          <div class="inst-vram-text"><span>VRAM</span><span>${fGiB(g.mem_used_mb)}/${fGiB(g.mem_total_mb)} GiB · ${num(g.mem_pct, 0)}%</span></div>
          <div class="vram-bar"><div class="vram-fill" style="width:${g.mem_pct}%"></div></div>
          <div style="margin-top:6px">${agg}</div>
        </div>
        <div class="inst-spark"><canvas data-spark="${g.uuid}"></canvas></div>
      </div>`;
  }).join("");
  renderSparklines();
}

function renderBanner() {
  const el = mq("#offload-banner");
  const o = MON.snap && MON.snap.ollama;
  if (o && o.spilled) {
    el.hidden = false;
    el.classList.toggle("warn", o.cpu_pct <= 5);
    el.textContent = `${o.model} is spilling to CPU — ${o.gpu_pct}% on GPU, ${o.cpu_pct}% on CPU. Throughput is degraded.`;
  } else {
    el.hidden = true;
  }
}

function renderNote() {
  const s = MON.snap;
  if (!s) return;
  const bits = [`sampling every ${s.interval_s}s`, `${s.retention_h}h history`];
  if (s.stale) bits.push("⚠ telemetry stale");
  if (s.error) bits.push("error: " + s.error);
  mq("#monitor-note").textContent = bits.join("  ·  ");
}

// ---- canvas charts (the signature element) -------------------------------
function setupCanvas(canvas, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 400;
  const h = cssH || canvas.clientHeight || 200;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

// series: [{points:[[ts,v]], color}]; opts: {yMin,yMax,redline,redlineColor,axes,now}
function drawChart(canvas, series, opts) {
  const o = opts || {};
  const { ctx, w, h } = setupCanvas(canvas, o.height);
  ctx.clearRect(0, 0, w, h);
  const padL = o.axes ? 38 : 4, padR = 8, padT = 8, padB = o.axes ? 18 : 4;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const all = series.flatMap((s) => s.points.map((p) => p[1]));
  let yMin = o.yMin != null ? o.yMin : Math.min(...all);
  let yMax = o.yMax != null ? o.yMax : Math.max(...all);
  if (!isFinite(yMin) || !isFinite(yMax)) { yMin = 0; yMax = 1; }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  if (o.padTop) yMax += (yMax - yMin) * o.padTop;

  const now = o.now || Date.now() / 1000;
  const xMin = now - MON.window, xMax = now;
  const X = (t) => padL + ((t - xMin) / (xMax - xMin)) * plotW;
  const Y = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  // grid + y labels
  ctx.strokeStyle = cssVar("--line");
  ctx.fillStyle = cssVar("--muted");
  ctx.lineWidth = 1;
  ctx.font = "10px " + cssVar("--mono");
  ctx.textBaseline = "middle";
  const rows = 4;
  for (let i = 0; i <= rows; i++) {
    const v = yMin + ((yMax - yMin) * i) / rows;
    const y = Math.round(Y(v)) + 0.5;
    ctx.globalAlpha = 0.35;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.globalAlpha = 1;
    if (o.axes) { ctx.textAlign = "right"; ctx.fillText(v.toFixed(o.yDec || 0), padL - 6, y); }
  }
  // x labels (time-ago)
  if (o.axes) {
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const frac = i / ticks, t = xMin + (xMax - xMin) * frac;
      const ago = (now - t) / 3600;
      const lbl = i === ticks ? "now" : "-" + (ago >= 1 ? ago.toFixed(0) + "h" : Math.round(ago * 60) + "m");
      ctx.fillText(lbl, X(t), h - padB + 5);
    }
  }

  // redline
  if (o.redline != null && o.redline >= yMin && o.redline <= yMax) {
    const y = Math.round(Y(o.redline)) + 0.5;
    ctx.strokeStyle = o.redlineColor || cssVar("--bad");
    ctx.globalAlpha = 0.5; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }

  // series lines
  ctx.lineWidth = 1.75; ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (const s of series) {
    if (!s.points.length) continue;
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    s.points.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
}

const METRICS = {
  temp: {
    title: (g) => g.name + " — temperature",
    series: (g) => [
      { points: g.series.temp, color: cssVar("--trace-core"), label: "core" },
      { points: g.series.hotspot, color: cssVar("--trace-hotspot"), label: "hotspot" },
      { points: g.series.junction, color: cssVar("--trace-junction"), label: "junction" },
    ],
    opts: () => ({ axes: true, yDec: 0, padTop: 0.1, redline: TEMP_REDLINE, redlineColor: cssVar("--trace-core") }),
  },
  power: {
    title: (g) => g.name + " — power draw",
    series: (g) => [{ points: g.series.power, color: cssVar("--trace-power"), label: "watts" }],
    opts: () => ({ axes: true, yMin: 0, yDec: 0, padTop: 0.1 }),
  },
  vram: {
    title: (g) => g.name + " — VRAM (GiB)",
    series: (g) => [{ points: g.series.vram.map((p) => [p[0], p[1] / 1024]), color: cssVar("--trace-vram"), label: "used" }],
    opts: (g) => ({ axes: true, yMin: 0, yMax: g.mem_total_mb / 1024, yDec: 0 }),
  },
};

function legend(items) {
  return `<div class="chart-legend">${items
    .map((it) => `<span><i style="background:${it.color}"></i>${it.label}</span>`)
    .join("")}</div>`;
}

function renderCharts() {
  const root = mq("#charts");
  const h = MON.hist;
  if (!h) { root.innerHTML = `<p class="chart-empty">Collecting history…</p>`; return; }

  if (MON.metric === "llm") {
    const gpu = h.ollama.gpu_pct, cpu = h.ollama.cpu_pct;
    const items = [
      { color: cssVar("--trace-gpu"), label: "on GPU %" },
      { color: cssVar("--trace-cpu"), label: "on CPU %" },
    ];
    root.innerHTML = `<div class="chart" style="grid-column:1/-1">
      <div class="chart-head"><span class="chart-title">Ollama GPU vs CPU split</span>${legend(items)}</div>
      ${gpu.length ? `<canvas data-llm="1"></canvas>` : `<p class="chart-empty">No Ollama model has been loaded in this window.</p>`}
    </div>`;
    const cv = root.querySelector("canvas[data-llm]");
    if (cv) drawChart(cv, [
      { points: gpu, color: cssVar("--trace-gpu") },
      { points: cpu, color: cssVar("--trace-cpu") },
    ], { axes: true, yMin: 0, yMax: 100, yDec: 0 });
    return;
  }

  const def = METRICS[MON.metric];
  root.innerHTML = h.gpus.map((g) => {
    const items = def.series(g).filter((s) => s.points.length).map((s) => ({ color: s.color, label: s.label }));
    const any = def.series(g).some((s) => s.points.length);
    return `<div class="chart">
      <div class="chart-head"><span class="chart-title">${def.title(g)}</span>${legend(items)}</div>
      ${any ? `<canvas data-uuid="${g.uuid}"></canvas>` : `<p class="chart-empty">No samples in this window.</p>`}
    </div>`;
  }).join("");
  h.gpus.forEach((g) => {
    const cv = root.querySelector(`canvas[data-uuid="${g.uuid}"]`);
    if (cv) drawChart(cv, def.series(g), def.opts(g));
  });
}

function renderSparklines() {
  if (!MON.hist) return;
  document.querySelectorAll("canvas[data-spark]").forEach((cv) => {
    const g = MON.hist.gpus.find((x) => x.uuid === cv.dataset.spark);
    if (!g || !g.series.temp.length) return;
    drawChart(cv, [{ points: g.series.temp, color: cssVar("--trace-core") }], { height: 46 });
  });
}

window.addEventListener("resize", () => { if (MON.active) { renderCharts(); renderSparklines(); } });
