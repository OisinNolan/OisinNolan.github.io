/* Etymap: English <-> Dutch etymology map.
   Data: data/index-<lang>.json, data/<lang>/<crc32 bucket>.json (words + form redirects),
         data/p/<crc32 bucket>.json (resolved pairs + nodes), data/langs.json, data/meta.json. */
"use strict";

const NOW = 2026;
const K = 500;                       // timeline scale: u ~ log(1 + (NOW - year) / K)
const EARLIEST = -4500;
let N_BUCKETS = 2048, W_BUCKETS = 256;   // overwritten from data/meta.json
const OTHER = { en: "nl", nl: "en" };
const NAME = { en: "English", nl: "Dutch" };
const EDGE_WORD = { inh: "inherited from", bor: "borrowed from", der: "derived from", cal: "calque of", cmp: "formed from" };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};

/* ---------- data access ---------- */

// Generated data is hosted on Hugging Face (too big for the site repo); local copies are used
// when the page is served from localhost.
const LOCAL = ["localhost", "127.0.0.1", ""].includes(location.hostname);
const DATA = LOCAL ? "data" : "https://oisinn-website-files.static.hf.space/etymap/data";

const cache = new Map();
function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).catch((e) => { cache.delete(path); throw e; }));
  }
  return cache.get(path);
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(str) {
  let c = 0xffffffff;
  for (const b of new TextEncoder().encode(str)) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
const bucketOf = (s, n = N_BUCKETS) => (crc32(s) % n).toString(16).padStart(3, "0");

async function lookupWord(lang, word) {
  const tryOne = async (w) => {
    let sh;
    try { sh = await getJSON(`${DATA}/${lang}/${bucketOf(w, W_BUCKETS)}.json`); } catch { return null; }
    if (sh.words[w]) return { word: w, cands: sh.words[w] };
    if (sh.forms[w]) {
      const lemma = sh.forms[w];
      const r = await tryOne(lemma);
      return r && { ...r, from: w };
    }
    return null;
  };
  return (await tryOne(word)) || (word !== word.toLowerCase() ? await tryOne(word.toLowerCase()) : null);
}

async function getPair(pid) {
  const b = await getJSON(`${DATA}/p/${bucketOf(pid)}.json`);
  return { pair: b.pairs[pid], nodes: b.nodes };
}

let LANGS = {};
const langP = Promise.all([getJSON(`${DATA}/langs.json`), getJSON(`${DATA}/meta.json`)])
  .then(([l, meta]) => { LANGS = l; N_BUCKETS = meta.buckets; W_BUCKETS = meta.word_buckets || W_BUCKETS; });

/* ---------- formatting ---------- */

const lang = (code) => LANGS[code] || { code, name: code };
const nodeLang = (n) => lang(n[1]);
function fmtYear(y) {
  y = Math.round(y);
  if (y >= NOW - 5) return "today";
  const r = Math.abs(y) >= 1000 ? Math.round(y / 50) * 50 : Math.round(y / 10) * 10;
  if (r < 0) return `${-r} BCE`;
  if (r < 1000) return `${r} CE`;
  return `${r}`;
}
function fmtRange(L) {
  if (L.from == null) return "";
  const a = L.from, b = L.to;
  const s = (y) => (y < 0 ? `${-y} BCE` : y < 1000 ? `${y} CE` : `${y}`);
  if (b >= NOW) return `${s(a)}–present`;
  return `c. ${s(a)} – ${s(b)}`;
}
function wiktionaryURL(n) {
  const L = nodeLang(n);
  let t = n[3] || n[0].replace(/^\*/, "");
  if (n[1].endsWith("-pro") && !t.startsWith("Reconstruction:")) t = `Reconstruction:${L.name}/${t.replace(/^\*/, "")}`;
  return `https://en.wiktionary.org/wiki/${encodeURIComponent(t.replace(/ /g, "_")).replace(/%2F/g, "/").replace(/%3A/g, ":")}#${encodeURIComponent(L.name.replace(/ /g, "_"))}`;
}

/* ---------- timeline scale ---------- */

const S = (y) => Math.log(1 + (NOW - y) / K);
let yMin = EARLIEST;
const yearToU = (y) => 1 - S(y) / S(yMin);
const uToYear = (u) => NOW - K * (Math.exp((1 - u) * S(yMin)) - 1);

/* ---------- model: turn a pair into drawable nodes/edges with times ---------- */

function buildModel(pair, nodes) {
  const info = new Map();   // key -> {key, n, side, t, parents: [{k, t, u, ax}], lca}
  const lcaSet = new Set(pair.lca || []);
  const enKeys = new Set(pair.A.en.map((r) => r[0]));
  const nlKeys = new Set(pair.A.nl.map((r) => r[0]));
  const shared = new Set();
  // shared = LCAs and everything above them
  const parentsOf = new Map();
  for (const side of ["en", "nl"]) for (const [k, , ps] of pair.A[side]) {
    const cur = parentsOf.get(k) || [];
    for (const p of ps) if (!cur.some((q) => q[0] === p[0])) cur.push(p);
    parentsOf.set(k, cur);
  }
  const stack = [...lcaSet];
  while (stack.length) {
    const k = stack.pop();
    if (shared.has(k)) continue;
    shared.add(k);
    for (const p of parentsOf.get(k) || []) stack.push(p[0]);
  }
  const add = (k, side) => {
    if (!info.has(k)) info.set(k, { key: k, n: nodes[k] || [k.split("|")[1], k.split("|")[0], "", ""], side, parents: [] });
    return info.get(k);
  };
  for (const side of ["en", "nl"]) for (const [k, d, ps] of pair.A[side]) {
    const s = shared.has(k) ? "sh" : (enKeys.has(k) && nlKeys.has(k) ? "sh" : side);
    const it = add(k, s);
    it.depth = Math.max(it.depth || 0, d);
    for (const [p, t, u, ax] of ps) {
      if (!it.parents.some((q) => q.k === p)) it.parents.push({ k: p, t, u: !!u, ax: !!ax });
      add(p, shared.has(p) ? "sh" : side);
    }
  }
  // ghost branches: chain from the branching node down into the other line
  const ghostEdges = [];
  for (const g of pair.ghosts || []) {
    const line = g.side === "nl" ? "en" : "nl";   // a ghost found on the Dutch path leads to English
    for (const br of g.branches) {
      let prev = g.from;
      for (const k of br) {
        const it = add(k, "ghost-" + line);
        if (!it.parents.some((q) => q.k === prev)) it.parents.push({ k: prev, t: "ghost", u: false, ghost: true });
        ghostEdges.push([prev, k]);
        prev = k;
      }
    }
  }
  for (const k of lcaSet) if (info.has(k)) info.get(k).lca = true;

  // times: language-stage start, forced monotonic along edges (parents first)
  const time = new Map();
  const visiting = new Set();
  const tOf = (k) => {
    if (time.has(k)) return time.get(k);
    if (visiting.has(k)) return lang(info.get(k).n[1]).from ?? 0;
    visiting.add(k);
    const it = info.get(k);
    const L = lang(it.n[1]);
    let t = L.from;
    const pt = it.parents.filter((p) => info.has(p.k)).map((p) => tOf(p.k));
    const maxP = pt.length ? Math.max(...pt) : -Infinity;
    if (t == null) t = pt.length ? maxP + 50 : null;
    else if (t <= maxP) t = maxP + 25;
    visiting.delete(k);
    time.set(k, t);
    return t;
  };
  for (const k of info.keys()) tOf(k);
  // nodes with unknown language and no parents: just before their earliest child
  for (const [k, it] of info) {
    if (time.get(k) != null) continue;
    const kids = [...info.values()].filter((c) => c.parents.some((p) => p.k === k)).map((c) => time.get(c.key)).filter((x) => x != null);
    time.set(k, kids.length ? Math.min(...kids) - 100 : 0);
  }
  for (const [k, it] of info) it.t = Math.min(NOW, Math.max(EARLIEST, time.get(k)));
  return { info, pair, nodes };
}

/* ---------- map ---------- */

const svg = d3.select("#map");
let projection, geoPath, landFeature, gLand, gHalos, gArcs, gNodes;
let W = 0, H = 0;
let zoomT = d3.zoomIdentity;
const zoom = d3.zoom()
  .scaleExtent([1, 24])
  .on("zoom", (e) => {
    zoomT = e.transform;
    if (gLand) gLand.attr("transform", zoomT);
    if (e.sourceEvent) {
      $("#maphint").classList.add("gone");
    }
    render(currentYear);
  });
svg.call(zoom);
const landP = getJSON("geo/land-50m.json").then((topo) => {
  landFeature = topojson.feature(topo, topo.objects.land);
});

function setupMap() {
  const box = svg.node().getBoundingClientRect();
  W = Math.max(320, box.width); H = Math.max(240, box.height);
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const pad = Math.min(70, W * 0.09);
  projection = d3.geoConicConformal().parallels([40, 62]).rotate([-14, 0])
    .fitExtent([[pad + 20, pad], [W - pad - 40, H - pad]], { type: "MultiPoint", coordinates: frameFor() });
  projection.clipExtent([[-W, -H], [2 * W, 2 * H]]);
  geoPath = d3.geoPath(projection);
  svg.selectAll("g, rect").remove();
  zoomT = d3.zoomIdentity;
  svg.property("__zoom", zoomT);
  zoom.extent([[0, 0], [W, H]]).translateExtent([[-W / 2, -H / 2], [1.5 * W, 1.5 * H]]);
  svg.append("rect").attr("class", "sea").attr("width", W).attr("height", H);
  gLand = svg.append("g");
  gLand.append("path").attr("class", "grat").attr("d", geoPath(d3.geoGraticule().step([10, 10])()));
  if (landFeature) gLand.append("path").attr("class", "land").attr("d", geoPath(landFeature));
  gHalos = svg.append("g");
  gArcs = svg.append("g");
  gNodes = svg.append("g");
  drawLens();
  updateZoomButtons();
}

function frameFor() {
  // bbox of the languages on screen, padded, with a minimum span; whole Europe when idle
  const pts = model ? [...model.info.values()].map((it) => lang(it.n[1])).filter((L) => L.lat != null && L.lon > -30 && L.lon < 70 && L.lat > 25) : [];
  if (!pts.length) return [[-11, 36], [44, 36], [-11, 63], [44, 63], [14, 66]];
  let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const L of pts) { x0 = Math.min(x0, L.lon); x1 = Math.max(x1, L.lon); y0 = Math.min(y0, L.lat); y1 = Math.max(y1, L.lat); }
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const hx = Math.max((x1 - x0) / 2 + 2, 7), hy = Math.max((y1 - y0) / 2 + 1.5, 4);
  return [[cx - hx, cy - hy], [cx + hx, cy - hy], [cx - hx, cy + hy], [cx + hx, cy + hy]];
}

function pxPerKm(lon, lat) {
  const a = projection([lon, lat]);
  const b = projection([lon + 1 / (111.32 * Math.cos((lat * Math.PI) / 180)), lat]);
  return a && b ? Math.hypot(b[0] - a[0], b[1] - a[1]) : 0;
}

/* zoom controls */
const NORTH_SEA = [[-5.8, 50.0], [7.6, 50.0], [-5.8, 55.7], [7.6, 55.7], [1, 56.2], [1, 49.4]];
function regionTransform(pts) {
  const xy = pts.map((p) => projection(p)).filter(Boolean);
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.max(1, Math.min(24, 0.92 * Math.min(W / (x1 - x0), H / (y1 - y0))));
  return d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
}
const zsel = (ms) => (reduceMotion ? svg : svg.transition().duration(ms));
const zoomTo = (t) => zsel(650).call(zoom.transform, t);
// Magnifier pinned near the England–Netherlands cluster. Clicking it zooms in and it turns into an
// "undo" button in the same spot, which returns to the previous view.
const LENS_AT = [1.6, 54.9];   // North Sea, just above the cluster
let nsK = 1;
let lens = { mode: "zoom", prev: null, base: null };   // base: spot in un-zoomed map coordinates
function updateZoomButtons() {
  nsK = projection ? regionTransform(NORTH_SEA).k : 1;
  lens = { mode: "zoom", prev: null, base: null };
}
function lensClick() {
  if (lens.mode === "zoom") {
    lens.mode = "undo";
    lens.prev = zoomT;
    zoomTo(regionTransform(NORTH_SEA));
  } else {
    lens.mode = "zoom";
    zoomTo(lens.prev || d3.zoomIdentity);
  }
}
function drawLens() {
  const g = svg.append("g").attr("id", "lens").attr("class", "lens").attr("tabindex", 0).attr("role", "button")
    .on("mousedown touchstart pointerdown", (e) => e.stopPropagation())
    .on("click", (e) => { e.stopPropagation(); lensClick(); })
    .on("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); lensClick(); } });
  g.append("title");
  g.append("circle").attr("class", "lensbg").attr("r", 17);
  const mag = g.append("g").attr("class", "ico-zoom");
  mag.append("circle").attr("cx", -2).attr("cy", -2).attr("r", 5.5);
  mag.append("path").attr("d", "M2.1,2.1L6.8,6.8");
  const undo = g.append("g").attr("class", "ico-undo");
  undo.append("path").attr("d", "M-6,-2.5H2.5A4.5,4.5 0 0 1 2.5,6.5H-3");
  undo.append("path").attr("d", "M-2.5,-6.5L-6.5,-2.5L-2.5,1.5");
}
// Choose the lens spot once per word, from the zoomed-out map with everything drawn: the first spot
// near LENS_AT clear of labels, dots and arcs. Afterwards it is pinned to the map like a label.
function chooseLensSpot(obstacles, arcPts) {
  const p = projection && projection(LENS_AT);
  if (!p) return null;
  const R = 19;
  const reserved = [[W - 62, 0, W, 150], [0, H - 120, 380, H]];   // zoom buttons, legend (screen space)
  const overlaps = (r, q) => r[0] < q[2] && r[2] > q[0] && r[1] < q[3] && r[3] > q[1];
  const onScreen = ([x, y]) => {   // fully visible and not under the controls
    const b = [x - R, y - R, x + R, y + R];
    return b[0] >= 6 && b[1] >= 6 && b[2] <= W - 6 && b[3] <= H - 6 && !reserved.some((q) => overlaps(b, q));
  };
  // the spot must also be visible (as the undo button) in the zoomed-in England & NL view
  const T = regionTransform(NORTH_SEA);
  const clear = (c) => {
    if (!onScreen(c) || !onScreen(T.apply(c))) return false;
    const b = [c[0] - R, c[1] - R, c[0] + R, c[1] + R];
    if (obstacles.some((q) => overlaps(b, q))) return false;
    return !arcPts.some(([px, py]) => px > b[0] - 3 && px < b[2] + 3 && py > b[1] - 3 && py < b[3] + 3);
  };
  for (let r = 0; r <= 90; r += 10) {
    for (let i = 0, n = r ? Math.round(r / 4) : 1; i < n; i++) {
      const a = (i / n) * 2 * Math.PI - Math.PI / 2;
      const c = [p[0] + r * Math.cos(a), p[1] + r * Math.sin(a)];
      if (clear(c)) return c;
    }
  }
  // nothing clear nearby: fall back to the anchor, pulled inside the zoomed-in view
  const [x0, y0] = T.invert([R + 12, R + 12]), [x1, y1] = T.invert([W - 70, H - 130]);
  return [Math.min(Math.max(p[0], x0), x1), Math.min(Math.max(p[1], y0), y1)];
}
function positionLens() {
  const g = svg.select("#lens");
  if (g.empty()) return;
  if (!model || !lens.base || nsK < 1.25) { g.style("display", "none"); return; }
  const x = zoomT.applyX(lens.base[0]), y = zoomT.applyY(lens.base[1]);
  const undo = lens.mode === "undo";
  g.style("display", x < 0 || y < 0 || x > W || y > H ? "none" : null)
    .attr("transform", `translate(${x},${y})`).classed("undo", undo)
    .attr("aria-label", undo ? "Back to the previous view" : "Zoom in on England and the Low Countries");
  g.select("title").text(undo ? "Back to the previous view" : "Zoom in on England and the Low Countries");
}
$("#zin").addEventListener("click", () => zsel(280).call(zoom.scaleBy, 1.7));
$("#zout").addEventListener("click", () => zsel(280).call(zoom.scaleBy, 1 / 1.7));
$("#zfit").addEventListener("click", () => zoomTo(d3.zoomIdentity));

let model = null;
let laid = null;   // key -> {ax, ay (projected anchor), ox, oy (pixel offset), r, it, lw, lh}

function layout() {
  laid = new Map();
  const groups = new Map();
  for (const it of model.info.values()) {
    const L = lang(it.n[1]);
    if (L.lat == null) continue;
    const p = projection([L.lon, L.lat]);
    if (!p || !isFinite(p[0])) continue;
    if (!groups.has(L.code)) groups.set(L.code, []);
    groups.get(L.code).push(it);
    it._p = p;
    it._r = Math.min(Math.max(10, (L.radius_km || 100) * pxPerKm(L.lon, L.lat)), 0.11 * Math.min(W, H * 1.6));
  }
  const rank = (it) => (it.side === "en" ? 0 : it.side === "sh" ? 1 : it.side === "nl" ? 2 : 3);
  for (const g of groups.values()) {
    g.sort((a, b) => rank(a) - rank(b) || a.t - b.t);
    g.forEach((it, i) => {
      // fan out multiple words of the same language around the anchor
      const ang = -Math.PI / 2 + i * 2.1;
      const rr = i === 0 ? 0 : 7 + 4 * i;
      laid.set(it.key, { ax: it._p[0], ay: it._p[1], ox: rr * Math.cos(ang), oy: rr * Math.sin(ang), r: it._r, it });
    });
  }
}
function screenPos(l) {
  const s = Math.sqrt(zoomT.k);   // spread same-language words a little more when zoomed in
  return [zoomT.applyX(l.ax) + l.ox * s, zoomT.applyY(l.ay) + l.oy * s];
}

function sideClass(it) { return it.lca ? "lca" : it.side; }
function arcClass(child, p) {
  const s = child.side.startsWith("ghost") ? `ghost ${child.side}` : child.side;
  return `arc ${s} ${p.ghost ? "ghost" : p.t}`;
}

function drawStatic() {
  setupMap();
  laid = null;
  if (!model) { $("#unmapped").hidden = true; render(currentYear); return; }
  layout();
  const haloLang = new Set();
  for (const l of laid.values()) {
    if (l.it.side.startsWith("ghost") || haloLang.has(l.it.n[1])) continue;
    haloLang.add(l.it.n[1]);
    gHalos.append("circle").attr("class", `halo ${sideClass(l.it)}`).datum(l).style("opacity", 0);
  }
  for (const l of laid.values()) for (const p of l.it.parents) {
    const a = laid.get(p.k);
    if (!a || p.ax) continue;
    gArcs.append("path").attr("class", arcClass(l.it, p))
      .datum({ a, b: l, ghost: !!p.ghost, t0: model.info.get(p.k)?.t ?? l.it.t, t1: l.it.t }).style("opacity", 0);
  }
  for (const l of laid.values()) {
    const it = l.it;
    const g = gNodes.append("g").attr("class", "node").attr("data-k", it.key).datum(l).style("opacity", 0)
      .on("click", () => focusNode(it.key))
      .on("mouseenter", () => showCard(it.key))
      .on("mouseleave", () => showCard(pinned));
    if (it.lca) g.append("circle").attr("class", "ring").attr("r", 8.5);
    g.append("circle").attr("class", `dot ${sideClass(it).replace(/^ghost-/, "")}`).attr("r", it.side.startsWith("ghost") ? 3.5 : 5);
    const t = g.append("text").attr("class", `lab${it.lca ? " lca" : ""}`).attr("x", 10).attr("y", 4);
    t.append("tspan").text(it.n[0]);
    t.append("tspan").attr("class", "ln").attr("dx", 5).text(nodeLang(it.n).name);
    l.lw = t.node().getBBox().width + 6;
    l.lh = 18;
    g.append("title").text(`${it.n[0]} (${nodeLang(it.n).name})${it.n[2] ? ` “${it.n[2]}”` : ""}`);
  }
  const un = [...model.info.values()].filter((it) => !laid.has(it.key)).map((it) => nodeLang(it.n).name);
  const box = $("#unmapped");
  box.hidden = !un.length;
  box.textContent = un.length ? `Not on the map: ${[...new Set(un)].join(", ")}` : "";
  lensPass = true;
  render(NOW);
  lensPass = false;
  render(currentYear);
}

function bezierPart({ a, c, b }, f) {
  const l = (p, q) => [p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f];
  const q0 = l(a, c), q1 = l(c, b), r = l(q0, q1);
  return `M${a[0]},${a[1]}Q${q0[0]},${q0[1]} ${r[0]},${r[1]}`;
}

let currentYear = NOW;
let lensPass = false;   // true for the one render that picks the lens spot
function render(year) {
  currentYear = year;
  $("#year").textContent = fmtYear(year);
  renderTree(year);
  if (!model || !laid || !gNodes) { positionLens(); return; }
  const u = yearToU(year);
  const P = new Map();
  for (const [k, l] of laid) P.set(k, screenPos(l));

  // the "current" form per line: the latest visible node
  const cur = {};
  for (const it of model.info.values()) {
    if (it.t > year) continue;
    if (!cur[it.side] || it.t > cur[it.side].t) cur[it.side] = it;
  }
  const curKeys = new Set(Object.values(cur).map((x) => x.key));
  const playingNow = year < NOW - 1;

  gHalos.selectAll("circle.halo").each(function (l) {
    const it = l.it, L = lang(it.n[1]), [x, y] = P.get(it.key);
    this.setAttribute("cx", x); this.setAttribute("cy", y); this.setAttribute("r", l.r * zoomT.k);
    const alive = it.t <= year && (L.to == null || year <= L.to + 50);
    this.style.opacity = it.t <= year ? (alive ? (L.certainty === "low" ? 0.07 : 0.11) : 0.04) : 0;
  });
  const arcPts = [];
  gArcs.selectAll("path").each(function (d) {
    const a = P.get(d.a.it.key), b = P.get(d.b.it.key);
    const dx = b[0] - a[0], dy = b[1] - a[1];
    if (Math.hypot(dx, dy) < 6) { this.style.opacity = 0; return; }
    const bend = 0.22 * (d.ghost ? -1 : 1);
    const c = [(a[0] + b[0]) / 2 - dy * bend, (a[1] + b[1]) / 2 + dx * bend];
    const u0 = yearToU(d.t0), u1 = yearToU(d.t1);
    const f = u1 <= u0 ? (u >= u1 ? 1 : 0) : Math.max(0, Math.min(1, (u - u0) / (u1 - u0)));
    this.style.opacity = f > 0 ? 1 : 0;
    this.setAttribute("d", bezierPart({ a, c, b }, Math.max(0.001, f)));
    for (let i = 0; f > 0 && i <= 24; i++) {
      const t = (i / 24) * f, m = 1 - t;
      arcPts.push([m * m * a[0] + 2 * m * t * c[0] + t * t * b[0], m * m * a[1] + 2 * m * t * c[1] + t * t * b[1]]);
    }
  });

  // greedy label placement (four-position model), most important labels first
  const prio = (it) => (it.depth === 0 && !it.side.startsWith("ghost") ? 0 : it.lca ? 1 : curKeys.has(it.key) && playingNow ? 2
    : it.side.startsWith("ghost") && it.side.slice(6) === it.n[1] ? 3 : it.side === "sh" ? 4 : 5);
  const placed = [];
  const dots = [...laid.values()].filter((l) => l.it.t <= year).map((l) => { const [x, y] = P.get(l.it.key); return [x - 5, y - 5, x + 5, y + 5]; });
  const overlaps = (r, q) => r[0] < q[2] && r[2] > q[0] && r[1] < q[3] && r[3] > q[1];
  const hit = (r) => placed.some((q) => overlaps(r, q)) || dots.some((q) => overlaps(r, q));
  const order = [...laid.values()].sort((a, b) => prio(a.it) - prio(b.it) || a.it.depth - b.it.depth);
  gNodes.selectAll("g.node").each(function (l) {
    const [x, y] = P.get(l.it.key);
    this.setAttribute("transform", `translate(${x},${y})`);
  });
  for (const l of order) {
    const it = l.it, k = it.key, [x, y] = P.get(k);
    const g = gNodes.select(`g.node[data-k="${CSS.escape(k)}"]`).node();
    if (!g) continue;
    const vis = it.t <= year;
    g.style.opacity = vis ? 1 : 0;
    g.style.pointerEvents = vis ? "" : "none";
    const lab = g.querySelector("text");
    lab.classList.toggle("cur", curKeys.has(k) && playingNow);
    lab.classList.toggle("dim", vis && !curKeys.has(k) && !it.lca && it.depth !== 0 && playingNow);
    if (!vis) { lab.style.display = "none"; continue; }
    const w = l.lw, h = l.lh;
    const opts = [[10, 4, "start", [x + 8, y - 11, x + 8 + w, y - 11 + h]],
                  [-10, 4, "end", [x - 8 - w, y - 11, x - 8, y - 11 + h]],
                  [0, -11, "middle", [x - w / 2, y - 26, x + w / 2, y - 26 + h]],
                  [0, 20, "middle", [x - w / 2, y + 6, x + w / 2, y + 6 + h]]]
      .filter((o) => o[3][0] >= 0 && o[3][2] <= W);
    const ok = opts.find((o) => !hit(o[3])) || (prio(it) <= 1 ? opts[0] : null);
    if (!ok) { lab.style.display = "none"; continue; }
    lab.style.display = "";
    lab.setAttribute("x", ok[0]); lab.setAttribute("y", ok[1]); lab.setAttribute("text-anchor", ok[2]);
    placed.push(ok[3]);
  }
  if (lensPass) lens.base = chooseLensSpot([...placed, ...dots.map((q) => [q[0] - 3, q[1] - 3, q[2] + 3, q[3] + 3])], arcPts);
  positionLens();
}

/* ---------- timeline ---------- */

const scrub = $("#scrub");
function drawTicks() {
  const svgT = d3.select("#ticks");
  const w = svgT.node().getBoundingClientRect().width || 600;
  svgT.attr("viewBox", `0 0 ${w} 30`).selectAll("*").remove();
  const x = (y) => 8 + yearToU(y) * (w - 16);   // range thumb inset
  const cands = [-4000, -3000, -2000, -1000, -500, 0, 500, 1000, 1500, 2000];
  let lastX = -99;
  for (const y of cands) {
    if (y < yMin) continue;
    const xx = x(y);
    if (xx - lastX < 58) continue;
    lastX = xx;
    svgT.append("line").attr("x1", xx).attr("x2", xx).attr("y1", 0).attr("y2", 5);
    svgT.append("text").attr("x", xx).attr("y", 17).text(y < 0 ? `${-y} BCE` : y === 0 ? "0" : `${y}`);
  }
  if (model) for (const it of model.info.values()) {
    if (it.side.startsWith("ghost")) continue;
    svgT.append("line").attr("class", `mk ${sideClass(it)}`).attr("x1", x(it.t)).attr("x2", x(it.t)).attr("y1", 22).attr("y2", 28)
      .append("title").text(`${it.n[0]} · ${nodeLang(it.n).name}`);
  }
}
scrub.addEventListener("input", () => { stop(); render(uToYear(scrub.value / 1000)); });

let playing = null;
function setPlayIcon(on) {
  $("#playicon").setAttribute("d", on ? "M4 2.5h3v11H4zM9 2.5h3v11H9z" : "M4 2.5v11l9-5.5z");
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
}
function stop() { if (playing) { cancelAnimationFrame(playing.raf); playing = null; setPlayIcon(false); } }
function play() {
  if (reduceMotion) { scrub.value = 1000; render(NOW); return; }
  stop();
  let u = scrub.value / 1000;
  if (u >= 0.999) u = 0;
  const dur = 9000;
  let last = performance.now();
  playing = {};
  setPlayIcon(true);
  const step = (now) => {
    u = Math.min(1, u + (now - last) / dur);
    last = now;
    scrub.value = Math.round(u * 1000);
    render(uToYear(u));
    if (u < 1) playing.raf = requestAnimationFrame(step);
    else stop();
  };
  playing.raf = requestAnimationFrame(step);
}
$("#play").addEventListener("click", () => (playing ? stop() : play()));

function setYear(y) {
  stop();
  y = Math.max(yMin, Math.min(NOW, y));
  scrub.value = Math.round(yearToU(y) * 1000);
  render(y);
}

function focusNode(k) {
  const it = model?.info.get(k);
  if (!it) return;
  setYear(Math.min(NOW, it.t + 1));
  pinned = k;
  showCard(k);
  const l = laid?.get(k);
  if (l) {
    const [x, y] = screenPos(l);
    if (x < 40 || y < 40 || x > W - 40 || y > H - 40) zoomTo(d3.zoomIdentity.translate(W / 2, H / 2).scale(zoomT.k).translate(-l.ax, -l.ay));
    const g = gNodes.select(`g.node[data-k="${CSS.escape(k)}"]`).node();
    if (g) { g.classList.remove("flash"); void g.getBBox(); g.classList.add("flash"); }
  }
  const tn = document.querySelector(`#tree g.tn[data-k="${CSS.escape(k)}"]`);
  if (tn) { tn.classList.remove("flash"); void tn.getBBox(); tn.classList.add("flash"); }
}

/* ---------- family tree (panel) ---------- */

const ABBR = {
  en: "English", nl: "Dutch", enm: "ME", ang: "OE", sco: "Scots", dum: "MDu", odt: "ODu",
  "gmw-pro": "PWGmc", "gem-pro": "PGmc", "ine-pro": "PIE", "itc-pro": "PItal", "cel-pro": "PCelt",
  "grk-pro": "PHell", "gmq-pro": "PNorse", non: "ON", fro: "OFr", frm: "MFr", fr: "French", xno: "AN",
  la: "Latin", grc: "AGk", gkm: "ByzGk", el: "Greek", osx: "OS", ofs: "OFris", fy: "Frisian", goh: "OHG",
  gmh: "MHG", gml: "MLG", nds: "LG", de: "German", got: "Gothic", frk: "Frank.", af: "Afrik.",
  it: "Italian", es: "Spanish", pt: "Port.", ar: "Arabic", fa: "Persian", he: "Hebrew", da: "Danish",
  sv: "Swedish", is: "Icel.", sga: "OIr", ga: "Irish", cy: "Welsh", pro: "OOcc",
};
function abbr(code) {
  if (ABBR[code]) return ABBR[code];
  const n = lang(code).name;
  return n.length <= 10 ? n : n.split(/[\s-]+/).map((w) => w[0]).join("");
}

function chainFrom(start) {
  // main line: follow the first non-affix parent
  const out = [];
  const seen = new Set();
  for (let k = start; k && !seen.has(k) && model.info.has(k);) {
    seen.add(k);
    out.push(k);
    const next = model.info.get(k).parents.find((p) => !p.ghost && !p.ax && model.info.has(p.k));
    k = next && next.k;
  }
  return out;
}
function pathTo(start, targets) {
  // breadth-first up the parents; returns [start … target] or null
  const prev = new Map([[start, null]]);
  const queue = [start];
  while (queue.length) {
    const k = queue.shift();
    if (targets.has(k)) { const out = []; for (let x = k; x != null; x = prev.get(x)) out.unshift(x); return out; }
    for (const p of model.info.get(k)?.parents || []) {
      if (p.ghost || p.ax || prev.has(p.k) || !model.info.has(p.k)) continue;
      prev.set(p.k, k);
      queue.push(p.k);
    }
  }
  return null;
}

let tree = null;     // {col, x, y, edges, knots, svg, height}
let pinned = null;   // key whose details the card shows when nothing is hovered

function layoutTree(width) {
  const { pair } = model;
  const lca = new Set(pair.lca || []);
  const col = new Map();
  const edges = [];
  const link = (child, parent, ghost) => {
    const p = model.info.get(child).parents.find((q) => q.k === parent);
    edges.push({ c: child, p: parent, t: ghost ? "ghost" : (p?.t || "inh"), u: !!p?.u, ghost });
  };
  const addChain = (keys, name) => keys.forEach((k, i) => {
    if (!col.has(k)) col.set(k, name);
    if (i + 1 < keys.length) link(k, keys[i + 1], false);
  });
  for (const l of lca) addChain(chainFrom(l), "sh");
  for (const side of ["en", "nl"]) addChain((lca.size && pathTo(pair[side], lca)) || chainFrom(pair[side]), side);
  for (const g of pair.ghosts || []) {
    if (!col.has(g.from)) continue;
    const name = `ghost-${g.side === "nl" ? "en" : "nl"}`;
    let prev = g.from;
    for (const k of g.branches[0]) {
      if (col.has(k)) break;
      col.set(k, name);
      link(k, prev, true);
      prev = k;
    }
  }
  // columns: a ghost track sits next to the line it branches from
  const used = new Set(col.values());
  const cols = ["en", "ghost-nl", "ghost-en", "nl"].filter((c) => c === "en" || c === "nl" || used.has(c));
  const labelW = Math.min(118, width * 0.32);
  const spacing = (width - 14 - labelW) / (cols.length - 1);
  const X = Object.fromEntries(cols.map((c, i) => [c, 14 + i * spacing]));
  X.sh = (X.en + X.nl) / 2;
  const maxW = Object.fromEntries(cols.map((c, i) => [c, i === cols.length - 1 ? width - X[c] - 12 : spacing - 16]));
  maxW.sh = width - X.sh - 12;

  // vertical time axis: the timeline's log scale, warped so that labels never collide
  const ts = [...new Set([...col.keys()].map((k) => model.info.get(k).t))].sort((a, b) => a - b);
  const top = 30, H0 = 560, gap = 50;
  const knots = [];
  for (const t of ts) {
    const u = yearToU(t);
    if (knots.length && u <= knots[knots.length - 1][0]) continue;
    knots.push([u, knots.length ? Math.max(top + u * H0, knots[knots.length - 1][1] + gap) : top + u * H0]);
  }
  if (!knots.length || knots[0][0] > 0) knots.unshift([0, (knots[0]?.[1] ?? top) - Math.min(18, top + (knots[0]?.[0] ?? 0) * H0)]);
  if (knots[knots.length - 1][0] < 1) knots.push([1, Math.max(top + H0, knots[knots.length - 1][1] + 28)]);
  return { col, X, maxW, edges, knots, height: knots[knots.length - 1][1] + 30 };
}

function interp(knots, v, from, to) {
  if (v <= knots[0][from]) return knots[0][to];
  for (let i = 1; i < knots.length; i++) {
    const a = knots[i - 1], b = knots[i];
    if (v <= b[from]) return a[to] + ((v - a[from]) / (b[from] - a[from] || 1)) * (b[to] - a[to]);
  }
  return knots[knots.length - 1][to];
}
const treeY = (year) => interp(tree.knots, yearToU(year), 0, 1);
const treeYear = (y) => uToYear(interp(tree.knots, y, 1, 0));

function drawTree() {
  const wrap = $("#treewrap");
  if (!wrap || !model) { tree = null; return; }
  const width = Math.max(260, wrap.getBoundingClientRect().width);
  tree = layoutTree(width);
  const s = d3.select("#tree").attr("viewBox", `0 0 ${width} ${tree.height}`).attr("width", width).attr("height", tree.height);
  s.selectAll("*").remove();
  const pos = (k) => [tree.X[tree.col.get(k)], treeY(model.info.get(k).t)];
  s.append("rect").attr("class", "tbg").attr("width", width).attr("height", tree.height);
  const lineOf = (c) => { const n = tree.col.get(c); return n.startsWith("ghost") ? `${n.slice(6)} ghost` : n; };
  const ge = s.append("g");
  for (const e of tree.edges) {
    const [x1, y1] = pos(e.p), [x2, y2] = pos(e.c);
    const ym = (y1 + y2) / 2;
    ge.append("path").attr("class", `te ${lineOf(e.c)} ${e.t}`).datum(e)
      .attr("d", x1 === x2 ? `M${x1},${y1}V${y2}` : `M${x1},${y1}C${x1},${ym} ${x2},${ym} ${x2},${y2}`);
  }
  const gn = s.append("g");
  for (const [k, c] of tree.col) {
    const it = model.info.get(k), [x, y] = pos(k);
    const g = gn.append("g").attr("class", `tn ${c.startsWith("ghost") ? "ghost" : ""}`).attr("data-k", k)
      .attr("transform", `translate(${x},${y})`).attr("tabindex", 0).attr("role", "button")
      .attr("aria-label", `${it.n[0]}, ${nodeLang(it.n).name}`)
      .on("click", () => focusNode(k))
      .on("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); focusNode(k); } })
      .on("mouseenter", () => showCard(k)).on("mouseleave", () => showCard(pinned));
    if (it.lca) g.append("circle").attr("class", "ring").attr("r", 8.5);
    const dotCls = it.lca ? "lca" : c.startsWith("ghost") ? c.slice(6) : c;
    g.append("circle").attr("class", `dot ${dotCls}`).attr("r", c.startsWith("ghost") ? 4 : 5.5);
    const f = g.append("text").attr("class", "tf").attr("x", 10).attr("y", -1).text(it.n[0]);
    const l = g.append("text").attr("class", "tl").attr("x", 10).attr("y", 11).text(abbr(it.n[1]));
    fit(f.node(), tree.maxW[c]);
    fit(l.node(), tree.maxW[c]);
    const gl = treeGloss(it);
    if (gl) fit(g.append("text").attr("class", "tg").attr("x", 10).attr("y", 24).text(`‘${gl}’`).node(), tree.maxW[c]);
    g.append("title").text(`${it.n[0]} · ${nodeLang(it.n).name}${it.n[2] ? ` · “${it.n[2]}”` : ""}`);
  }
  const yl = s.append("g").attr("class", "yl");
  yl.append("line").attr("x1", 0).attr("x2", width);
  yl.append("rect").attr("class", "ylbg").attr("y", -9).attr("height", 18).attr("rx", 9);
  yl.append("text").attr("class", "ylt").attr("y", 4);

  // drag on the tree to scrub through time
  const svgEl = s.node();
  const toYear = (ev) => treeYear(ev.clientY - svgEl.getBoundingClientRect().top);
  svgEl.onpointerdown = (ev) => {
    if (ev.target.closest(".tn")) return;
    if (ev.pointerType === "touch") {   // touch: keep page scrolling; a tap jumps to that time
      const y0 = ev.clientY;
      svgEl.onpointerup = (e2) => { if (Math.abs(e2.clientY - y0) < 8) setYear(toYear(e2)); svgEl.onpointerup = null; };
      return;
    }
    ev.preventDefault();   // no text selection while dragging
    svgEl.setPointerCapture(ev.pointerId);
    setYear(toYear(ev));
    svgEl.onpointermove = (e2) => setYear(toYear(e2));
  };
  svgEl.addEventListener("pointerup", () => { svgEl.onpointermove = null; });
  svgEl.addEventListener("pointercancel", () => { svgEl.onpointermove = null; });
  renderTree(currentYear);
}

// the meaning at the time, in English: first sense of the Wiktionary gloss (not for modern English
// words, whose "gloss" is a definition rather than a translation)
function treeGloss(it) {
  const g = (it.n[2] || "").replace(/\[[^\]]*\]?|\([^)]*\)?/g, "").trim();
  if (!g || it.n[1] === "en") return "";
  const first = g.split(/[,;]/)[0].replace(/[“”"]/g, "").replace(/[.…]+$/, "").trim();
  return first.toLowerCase() === it.n[0].replace(/^\*/, "").toLowerCase() ? "" : first;
}

function fit(textEl, maxW) {
  let s = textEl.textContent;
  while (s.length > 2 && textEl.getComputedTextLength() > maxW) {
    s = s.slice(0, -1);
    textEl.textContent = s + "…";
  }
}

function renderTree(year) {
  if (!tree || !model) return;
  const s = d3.select("#tree");
  const y = treeY(year);
  const yl = s.select("g.yl").attr("transform", `translate(0,${y})`);
  const label = fmtYear(year);
  const txt = yl.select("text").text(label);
  const tw = txt.node().getComputedTextLength();
  const width = +s.attr("width");
  txt.attr("x", width - 10).attr("text-anchor", "end");
  yl.select("rect").attr("x", width - tw - 18).attr("width", tw + 16);
  const cur = {};
  for (const [k, c] of tree.col) {
    const it = model.info.get(k);
    if (it.t <= year && (!cur[c] || it.t > model.info.get(cur[c]).t)) cur[c] = k;
  }
  const curKeys = new Set(Object.values(cur));
  s.selectAll("g.tn").each(function () {
    const k = this.getAttribute("data-k");
    const it = model.info.get(k);
    this.classList.toggle("future", it.t > year);
    this.classList.toggle("cur", curKeys.has(k) && year < NOW - 1);
  });
  s.selectAll("path.te").each(function (e) {
    this.classList.toggle("future", model.info.get(e.c).t > year);
  });
  // keep the "now" line in view while playing
  const panel = $("#panel");
  if (playing && panel.scrollHeight > panel.clientHeight + 4 && getComputedStyle(panel).overflowY !== "visible") {
    const off = s.node().getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop;
    panel.scrollTop = Math.max(0, off + y - panel.clientHeight * 0.45);
  }
}

function showCard(k) {
  const card = $("#card");
  const it = k && model?.info.get(k);
  if (!card || !it) return;
  const L = nodeLang(it.n);
  const ps = it.parents.filter((p) => !p.ghost);
  const main = ps.find((p) => !p.ax);
  const ghostIn = it.parents.find((p) => p.ghost);
  const range = fmtRange(L);
  const name = (key) => { const n = model.info.get(key)?.n; return n ? [el("i", {}, n[0]), ` (${nodeLang(n).name})`] : []; };
  card.replaceChildren(...[
    el("div", { class: "cf" },
      el("span", { class: `form ${it.side.replace("ghost-", "")}` }, it.n[0]), " ",
      el("span", { class: "lang" }, L.name),
      it.lca ? el("span", { class: "badge" }, "common ancestor") : null,
      el("a", { class: "wk", href: wiktionaryURL(it.n), target: "_blank", rel: "noopener" }, "Wiktionary ↗")),
    it.n[2] || range ? el("div", { class: "cg" }, it.n[2] ? `“${it.n[2]}”` : "", it.n[2] && range ? " · " : "", range) : null,
    ps.length > 1 ? el("div", { class: "cv" }, "= ", ...ps.flatMap((p, i) => [i ? " + " : "", el("i", {}, model.info.get(p.k)?.n[0] || p.k.split("|")[1])])) : null,
    main && main.t !== "cmp" ? el("div", { class: "cv" }, `${EDGE_WORD[main.t] || "from"} `, ...name(main.k),
      main.u ? el("span", { class: "unc", title: "Wiktionary marks this link as uncertain" }, " ?") : null) : null,
    ghostIn && !main ? el("div", { class: "cv" }, "from ", ...name(ghostIn.k)) : null,
  ].filter(Boolean));
}

function renderPanel(state) {
  const panel = $("#panel");
  panel.textContent = "";
  panel.scrollTop = 0;
  const { pair } = model;
  panel.append(
    el("h2", {}, "Family tree"),
    el("p", { class: "thint" }, "Drag up or down on the tree to travel through time. Click a word for details."),
    el("div", { id: "treewrap" }, document.createElementNS("http://www.w3.org/2000/svg", "svg")),
  );
  $("#treewrap svg").id = "tree";
  if (pair.ghosts?.length) {
    panel.append(el("h2", {}, "Hidden relatives"));
    for (const g of pair.ghosts) {
      const line = g.side === "nl" ? "en" : "nl";
      const from = model.info.get(g.from);
      const leaves = g.branches.map((b) => model.info.get(b[b.length - 1]));
      const modern = leaves.filter((x) => x.n[1] === line);
      const wordBtn = (k) => el("button", { type: "button", class: "wl", onclick: () => focusNode(k) }, model.info.get(k).n[0]);
      const box = el("div", { class: "ghostbox" });
      box.append(el("p", {},
        `${NAME[g.side]} `, el("i", {}, model.info.get(pair[g.side]).n[0]), " goes back to ", wordBtn(g.from),
        ` (${nodeLang(from.n).name}), which also gave ${NAME[line]} `,
        ...(modern.length ? modern.flatMap((x, i) => [i ? (i === modern.length - 1 ? " and " : ", ") : "", el("b", {}, wordBtn(x.key))]) : ["words that have since died out"]),
        modern.length ? "." : el("span", { class: "dead" }, " †")));
      for (const b of g.branches) {
        box.append(el("p", { class: `route ${line}` }, ...b.flatMap((k, i) => [i ? " → " : "",
          el("span", { class: "rl" }, abbr(model.info.get(k).n[1])), " ", wordBtn(k)]),
          model.info.get(b[b.length - 1]).n[1] !== line ? el("span", { class: "dead" }, " †") : ""));
      }
      panel.append(box);
    }
  }
  panel.append(el("div", { class: "card", id: "card", "aria-live": "polite" }));
  pinned = pair[state.dir];
  drawTree();
  showCard(pinned);
}

/* ---------- header ---------- */

function summary(state) {
  const { pair } = model;
  const w = (side) => el("em", {}, model.info.get(pair[side]).n[0]);
  const parts = [];
  const flags = pair.flags || {};
  const lcaK = pair.lca?.[0];
  if (flags.loan === "en<nl") parts.push("English ", w("en"), " was borrowed from Dutch ", w("nl"), ".");
  else if (flags.loan === "nl<en") parts.push("Dutch ", w("nl"), " was borrowed from English ", w("en"), ".");
  else if (lcaK) {
    const it = model.info.get(lcaK);
    const L = nodeLang(it.n);
    const anyBor = [...model.info.values()].some((x) => !x.side.startsWith("ghost") && x.parents.some((p) => p.t === "bor"));
    parts.push(anyBor ? "Both go back to " : "Cognates: both descend from ", L.name, " ", el("em", {}, it.n[0]),
      L.from != null ? ` (${fmtRange(L)})` : "", anyBor ? ", by different routes." : ".");
  } else if (pair.ghosts?.length) {
    const g = pair.ghosts[0];
    const line = g.side === "nl" ? "en" : "nl";
    const leaf = g.branches.map((b) => model.info.get(b[b.length - 1])).find((x) => x.n[1] === line);
    parts.push("No common ancestor: these are unrelated words. ");
    if (leaf) parts.push("But ", NAME[g.side], " ", w(g.side), " has an ", NAME[line], " cousin: ", el("em", {}, leaf.n[0]), ".");
    else parts.push(NAME[g.side], " ", w(g.side), " once had an ", NAME[line], " relative, but that branch died out.");
  } else {
    parts.push("No common ancestor found in Wiktionary's data.");
  }
  const notes = [];
  for (const side of ["en", "nl"]) {
    const f = flags[side] || [];
    if (f.includes("unknown")) notes.push(`The origin of ${model.info.get(pair[side]).n[0]} is unknown.`);
    else if (f.includes("uncertain")) notes.push(`The origin of ${model.info.get(pair[side]).n[0]} is uncertain.`);
    else if (f.includes("alt_chain")) notes.push(`Wiktionary gives more than one possible history for ${model.info.get(pair[side]).n[0]}; this shows the first.`);
  }
  return [el("p", { class: "summary" }, ...parts), ...notes.map((n) => el("p", { class: "note" }, n))];
}

function renderHead(state, res) {
  const head = $("#head");
  head.textContent = "";
  const { dir, cands, sel } = state;
  const c = cands[sel];
  head.append(el("h2", { class: "title" },
    el("span", { class: `w ${dir}` }, res.word),
    el("span", { class: "to" }, "→"),
    el("span", { class: `w ${OTHER[dir]}` }, c[0])));
  if (res.from) head.append(el("p", { class: "note" }, `Showing ${res.word} (you typed ${res.from}).`));
  if (cands.length > 1) {
    const chips = el("div", { class: "chips", role: "group", "aria-label": "Other translations" });
    cands.forEach((cc, i) => chips.append(el("button", {
      type: "button", class: `chip ${OTHER[dir]}`, "aria-pressed": i === sel ? "true" : "false",
      disabled: !cc[3], title: [cc[1], cc[2]].filter(Boolean).join(" · ") + (cc[3] ? "" : " (not traced)"),
      onclick: () => { location.hash = `#${dir}/${encodeURIComponent(res.word)}/${encodeURIComponent(cc[0])}`; },
    }, cc[0])));
    head.append(chips);
  }
  if (c[1]) head.append(el("p", { class: "note" }, `sense: ${c[1]}`));
  if (model) head.append(...summary(state));
}

function renderMessage(html) {
  model = null;
  tree = null;
  $("#head").innerHTML = html;
  $("#panel").textContent = "";
  drawStatic();
  drawTicks();
}

/* ---------- routing ---------- */

let state = { dir: "en" };
function setDir(dir) {
  state.dir = dir;
  $("#dir").classList.toggle("rev", dir === "nl");
  $("#dir [data-from]").textContent = dir.toUpperCase();
  $("#dir [data-to]").textContent = OTHER[dir].toUpperCase();
  $("#q").placeholder = dir === "en" ? "Type an English word…" : "Typ een Nederlands woord…";
}

async function route() {
  const [dir, word, pick] = location.hash.slice(1).split("/").map((s) => decodeURIComponent(s || ""));
  if (!word || !["en", "nl"].includes(dir)) { setDir(dir === "nl" ? "nl" : "en"); return; }
  setDir(dir);
  $("#q").value = word;
  await langP;
  const res = await lookupWord(dir, word.trim());
  if (!res) {
    renderMessage(`<p class="notfound">No ${NAME[dir]} entry for <em>${escapeHTML(word)}</em> with a ${NAME[OTHER[dir]]} translation in Wiktionary's data.</p>
      <p class="note">Try the dictionary form (e.g. <em>take</em> rather than <em>took</em>), or switch direction.</p>`);
    return;
  }
  const cands = res.cands;
  let sel = pick ? cands.findIndex((c) => c[0] === pick) : -1;
  if (sel < 0) sel = Math.max(0, cands.findIndex((c) => c[3]));
  state = { dir, word: res.word, cands, sel };
  const pid = cands[sel][3];
  if (!pid) {
    model = null;
    tree = null;
    renderHead(state, res);
    $("#head").append(el("p", { class: "summary" }, "This translation is a phrase or wasn't traced, so there is no history to show."));
    $("#panel").textContent = "";
    drawStatic(); drawTicks();
    return;
  }
  let got;
  try { got = await getPair(pid); } catch {
    renderMessage(`<p class="notfound">Couldn't load the data for this word. Please try again.</p>`);
    return;
  }
  const { pair, nodes } = got;
  await landP;
  model = buildModel(pair, nodes);
  const ts = [...model.info.values()].map((x) => x.t);
  yMin = Math.max(EARLIEST, Math.min(...ts) - 150);
  renderHead(state, res);
  renderPanel(state);
  drawStatic();
  drawTicks();
  scrub.value = 0;
  if (reduceMotion) { scrub.value = 1000; render(NOW); } else { render(uToYear(0)); play(); }
}

function escapeHTML(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

/* ---------- search + autocomplete ---------- */

const q = $("#q"), list = $("#suggest");
let active = -1;
async function suggest() {
  const v = q.value.trim().toLowerCase();
  if (!v) { closeList(); return; }
  let idx;
  try { idx = await getJSON(`${DATA}/index-${state.dir}.json`); } catch { return; }
  if (q.value.trim().toLowerCase() !== v) return;
  // binary search on the lowercase-sorted list
  let lo = 0, hi = idx.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (idx[m].toLowerCase() < v) lo = m + 1; else hi = m; }
  const out = [];
  for (let i = lo; i < idx.length && out.length < 8 && idx[i].toLowerCase().startsWith(v); i++) out.push(idx[i]);
  list.textContent = "";
  out.forEach((w, i) => list.append(el("li", { role: "option", id: `s${i}`, "aria-selected": "false",
    onmousedown: (e) => { e.preventDefault(); go(w); } }, w)));
  active = -1;
  list.hidden = !out.length;
  q.setAttribute("aria-expanded", out.length ? "true" : "false");
}
function closeList() { list.hidden = true; q.setAttribute("aria-expanded", "false"); active = -1; }
function go(w) { closeList(); q.blur(); location.hash = `#${state.dir}/${encodeURIComponent(w.trim())}`; }
q.addEventListener("input", suggest);
q.addEventListener("focus", () => getJSON(`${DATA}/index-${state.dir}.json`).catch(() => {}));
q.addEventListener("blur", () => setTimeout(closeList, 100));
q.addEventListener("keydown", (e) => {
  const items = [...list.children];
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    if (!items.length) return;
    e.preventDefault();
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
    items.forEach((li, i) => li.setAttribute("aria-selected", i === active ? "true" : "false"));
    q.setAttribute("aria-activedescendant", `s${active}`);
  } else if (e.key === "Escape") closeList();
});
$("#search").addEventListener("submit", (e) => {
  e.preventDefault();
  const items = [...list.children];
  go(active >= 0 && items[active] ? items[active].textContent : q.value);
});
$("#dir").addEventListener("click", () => {
  // swap direction; if a pair is showing, look the translation up the other way
  if (state.word && state.cands) {
    const other = state.cands[state.sel][0];
    location.hash = `#${OTHER[state.dir]}/${encodeURIComponent(other)}/${encodeURIComponent(state.word)}`;
  } else {
    setDir(OTHER[state.dir]);
    q.value = "";
    q.focus();
  }
});

/* ---------- boot ---------- */

window.addEventListener("hashchange", route);
let resizeT;
new ResizeObserver(() => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => {
    const b = svg.node().getBoundingClientRect();
    if (Math.abs(Math.max(320, b.width) - W) < 2 && Math.abs(Math.max(240, b.height) - H) < 2) return;
    drawStatic(); drawTicks();
  }, 120);
}).observe($(".stage"));
let treeW = 0, treeT;
new ResizeObserver(() => {
  const w = $("#panel").clientWidth;
  if (Math.abs(w - treeW) < 2) return;
  treeW = w;
  clearTimeout(treeT);
  treeT = setTimeout(() => { if (model) { drawTree(); showCard(pinned); } }, 120);
}).observe($("#panel"));
Promise.all([landP, langP]).then(() => { drawStatic(); drawTicks(); render(NOW); route(); });
