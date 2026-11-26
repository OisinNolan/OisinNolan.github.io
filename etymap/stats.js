/* Etymap stats page: renders stats.json (built by _etymap/build/stats.py). */
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style") e.setAttribute("style", v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};
const fmt = (n) => n.toLocaleString("en-GB");
const pct = (n, d) => (d ? (100 * n) / d : 0);
const pctStr = (p) => (p >= 10 ? p.toFixed(0) : p >= 1 ? p.toFixed(1) : p.toFixed(2)) + "%";
const link = (lang, w, other) => el("a", { class: "w", href: `./#${lang}/${encodeURIComponent(w)}${other ? "/" + encodeURIComponent(other) : ""}` }, w);
const NAME = { en: "English", nl: "Dutch" };
const NATIVE = new Set(["gmw-pro", "gem-pro", "ine-pro", "frk"]);

/* tooltip */
const tip = $("#tip");
function hoverTip(node, html) {
  const show = (e) => { tip.replaceChildren(...html()); tip.classList.add("on"); move(e); };
  const move = (e) => {
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(innerWidth - r.width - 8, e.clientX + 14) + "px";
    tip.style.top = Math.max(8, e.clientY - r.height - 12) + "px";
  };
  node.addEventListener("mouseenter", show);
  node.addEventListener("mousemove", move);
  node.addEventListener("mouseleave", () => tip.classList.remove("on"));
}

/* one bar row: label | bar + value, optional examples line underneath */
function barRow({ label, sub, n, total, max, cls = "", examples, tipText }) {
  const p = pct(n, total);
  const bar = el("div", { class: `bar ${cls}`, style: `width:${Math.max(0.4, (100 * n) / max)}%` });
  const row = el("div", { class: "row" },
    el("div", { class: "lab" }, label, sub ? el("small", {}, sub) : null),
    el("div", { class: "track" }, bar, el("span", { class: "val" }, el("b", {}, pctStr(p)), " ", el("span", {}, `(${fmt(n)})`))),
    examples && examples.length ? el("div", { class: "ex" }, ...examples) : null);
  if (tipText) hoverTip(row, tipText);
  return row;
}
const exList = (items) => items.flatMap((x, i) => [i ? ", " : "", x]);

fetch("stats.json").then((r) => r.json()).then(render).catch(() => {
  document.body.append(el("p", { class: "note" }, "Couldn't load the statistics."));
});

function render(S) {
  /* tiles */
  const kn = Object.fromEntries(S.kinds.map((k) => [k.kind, k.n]));
  const related = (kn.cognate || 0) + (kn.cognate_routes || 0) + (kn.loan_en || 0) + (kn.loan_nl || 0);
  const withData = S.kinds.reduce((a, k) => a + k.n, 0) - (kn.no_data || 0);
  $("#tiles").append(
    el("div", { class: "tile" }, el("div", { class: "v" }, fmt(S.counts.en_words)), el("div", { class: "l" }, "English words with a Dutch translation")),
    el("div", { class: "tile" }, el("div", { class: "v" }, fmt(S.counts.nl_words)), el("div", { class: "l" }, "Dutch words with an English translation")),
    el("div", { class: "tile" }, el("div", { class: "v" }, pctStr(pct(related, withData))),
      el("div", { class: "l" }, "of English words share an ancestor with their Dutch translation, where both histories are known")));

  /* 0. cognate finder */
  if (S.cognates) {
    const C = S.cognates.bands;
    $("#cogtiles").append(
      el("div", { class: "tile" }, el("div", { class: "v" }, pctStr(pct(C.nl.with, C.nl.words))), el("div", { class: "l" }, `of ${fmt(C.nl.words)} Dutch words have at least one English relative`)),
      el("div", { class: "tile" }, el("div", { class: "v" }, pctStr(pct(C.en.with, C.en.words))), el("div", { class: "l" }, `of ${fmt(C.en.words)} English words have at least one Dutch relative`)),
      el("div", { class: "tile" }, el("div", { class: "v" }, pctStr(pct(C.nl.with_pie, C.nl.with))), el("div", { class: "l" }, "of Dutch words with a relative also have a distant one, through Proto-Indo-European")));
    const BLAB = { loan: ["Borrowed directly", "one language took the word from the other"],
      west: ["Proto-West Germanic", "c. 200 BCE – 500 CE"], gem: ["Proto-Germanic", "c. 500 BCE – 200 CE"],
      classical: ["Latin, Greek or French", "both borrowed from the same source"], pie: ["Proto-Indo-European", "c. 4500–2500 BCE"],
      other: ["Elsewhere", "other languages"] };
    for (const side of ["nl", "en"]) {
      const B = C[side], other = side === "nl" ? "en" : "nl";
      const box = el("div", {}, el("h3", {}, el("i", { class: `k ${side}` }), `${NAME[side]} words`));
      const m = Math.max(...B.by.map((x) => x.n));
      for (const x of B.by) {
        const [label, sub] = BLAB[x.band];
        box.append(barRow({ label, sub, n: x.n, total: B.with, max: m, cls: x.band === "pie" || x.band === "west" || x.band === "gem" ? "" : "alt",
          examples: exList(x.examples.slice(0, 3).map(([w, c]) => el("span", {}, link(side, w, c), " ~ ", el("span", { class: "w" }, c)))) }));
      }
      $("#cogbands").append(box);
    }
    const tb = $("#far tbody");
    for (const [w, near, far, via] of S.cognates.far.nl.slice(0, 12)) {
      const to = (c) => el("a", { class: "w", href: `./#nl/${encodeURIComponent(w)}/${encodeURIComponent(c)}` }, c);
      tb.append(el("tr", {}, el("td", {}, link("nl", w)), el("td", {}, to(near)), el("td", {}, to(far)),
        el("td", {}, el("span", { class: "w" }, via))));
    }
  }

  /* 1. kinds */
  const KIND = {
    cognate: ["Cognates", "both inherited from a shared ancestor"],
    cognate_routes: ["Common ancestor, but borrowed along the way", "e.g. both from Latin, by different routes"],
    loan_en: ["English borrowed it from Dutch", null],
    loan_nl: ["Dutch borrowed it from English", null],
    cousin: ["Unrelated, but one has a cognate in the other language", "e.g. Dutch hond's English cognate is hound, not dog"],
    dead_cousin: ["Unrelated; the cognate has died out", "it existed only in Old or Middle English"],
    unrelated: ["Unrelated", "no link found"],
    no_data: ["Not enough data", "one of the two words has no etymology on Wiktionary"],
  };
  const total = S.kinds.reduce((a, k) => a + k.n, 0);
  const max = Math.max(...S.kinds.map((k) => k.n));
  for (const k of S.kinds) {
    const [label, sub] = KIND[k.kind];
    $("#kinds").append(barRow({
      label, sub, n: k.n, total, max, cls: k.kind === "no_data" ? "alt" : "",
      examples: exList(k.examples.slice(0, 5).map(([e, d]) => el("span", {}, link("en", e, d), " / ", el("span", { class: "w" }, d)))),
    }));
  }

  /* 2. where they meet: top stages, oldest first */
  const withDate = S.lca.filter((x) => x.from != null);
  const top = withDate.slice().sort((a, b) => b.n - a.n).slice(0, 11).sort((a, b) => a.from - b.from || b.n - a.n);
  const rest = S.lca.reduce((a, x) => a + x.n, 0) - top.reduce((a, x) => a + x.n, 0);
  const lcaTotal = S.lca.reduce((a, x) => a + x.n, 0);
  const lcaMax = Math.max(...top.map((x) => x.n), rest);
  const yr = (y) => (y < 0 ? `${-y} BCE` : y < 1000 ? `${y} CE` : `${y}`);
  for (const x of top) {
    $("#lca").append(barRow({
      label: x.name, sub: x.to >= 2026 ? `${yr(x.from)}–present` : `c. ${yr(x.from)} – ${yr(x.to)}`,
      n: x.n, total: lcaTotal, max: lcaMax, cls: NATIVE.has(x.code) ? "" : "alt",
      examples: exList(x.examples.slice(0, 4).map(([e, d, a]) => el("span", {}, link("en", e, d), " / ", el("span", { class: "w" }, d), " ← ", el("span", { class: "w" }, a)))),
    }));
  }
  if (rest > 0) $("#lca").append(barRow({ label: "Other", sub: "smaller groups", n: rest, total: lcaTotal, max: lcaMax, cls: "alt" }));

  /* 3a. real replacements */
  const RP = S.replaced.replaced, BL = S.replaced.built, cg = RP.cognate;
  $("#replaced-intro").append(
    "Pairs where the Dutch word was inherited from Proto-West Germanic or older, so the ancestor of English had the word too, ",
    "but English now uses a loanword: ", el("b", {}, fmt(RP.total)), " pairs. In ", fmt(cg.replaced),
    " of them Etymap found English's own cognate. In ", fmt(cg.survives), " it still exists, usually with a narrower or different meaning (Dutch ",
    el("span", { class: "w" }, "taal"), " “language” and English ", el("span", { class: "w" }, "tale"), "; Dutch ",
    el("span", { class: "w" }, "hemel"), " “sky” and English ", el("span", { class: "w" }, "heaven"), "), and in ", fmt(cg.died),
    " it has died out. These are the languages English took the replacement words from:");
  const byBars = (target, R) => {
    const m = Math.max(...R.by.map((x) => x.n));
    for (const x of R.by.filter((x) => x.family !== "Other").concat(R.by.filter((x) => x.family === "Other"))) {
      $(target).append(barRow({ label: x.family, n: x.n, total: R.total, max: m, cls: x.family === "Other" ? "alt" : "",
        examples: target === "#built" ? exList(x.examples.slice(0, 5).map(([e, d]) => el("span", {}, link("en", e, d), " / ", el("span", { class: "w" }, d)))) : null }));
    }
  };
  byBars("#replaced", RP);
  const tb = $("#replaced-table tbody");
  for (const x of RP.by.filter((x) => x.family !== "Other").slice(0, 4)) {
    for (const [e, d, c, dead] of x.examples.slice(0, x.family === "French" ? 4 : 3)) {
      tb.append(el("tr", {},
        el("td", {}, link("en", e, d), el("span", { class: "dead" }, ` (${x.family})`)),
        el("td", {}, link("nl", d, e)),
        el("td", {}, c ? (dead ? el("span", { class: "dead" }, el("span", { class: "w" }, c), " †") : el("span", { class: "w" }, c)) : el("span", { class: "dead" }, "not found"))));
    }
  }

  /* 3b. built vs borrowed */
  $("#built-intro").append(
    "Pairs where Dutch formed the word itself, by compounding or deriving from native parts, while English borrowed one: ",
    el("b", {}, fmt(BL.total)), " pairs. These are not replacements: the concept often arrived later, and the two languages ",
    "handled it differently. Dutch ", el("span", { class: "w" }, "woordenboek"), " is literally “word book”, where English has ",
    el("span", { class: "w" }, "dictionary"), "; ", el("span", { class: "w" }, "neushoorn"), " is “nose horn” for ",
    el("span", { class: "w" }, "rhinoceros"), ". Where English borrowed its word from:");
  byBars("#built", BL);

  /* 4. fingerprints: same rows, same scale, side by side */
  const ORDER = ["Native Germanic", "French", "Latin", "Greek", "Norse", "German", "OTHER_LANG", "Low German & Frisian", "Other Romance", "Other"];
  const known = {};
  for (const side of ["en", "nl"]) {
    const list = S.fingerprint[side].filter((x) => x.family !== "No etymology");
    known[side] = { total: list.reduce((a, x) => a + x.n, 0), by: Object.fromEntries(list.map((x) => [x.family, x])) };
  }
  const famOf = (side, f) => (f === "OTHER_LANG" ? (side === "en" ? "Dutch" : "English") : f);
  const fpMax = Math.max(...["en", "nl"].flatMap((s) => ORDER.map((f) => pct(known[s].by[famOf(s, f)]?.n || 0, known[s].total))));
  for (const side of ["en", "nl"]) {
    const box = el("div", {}, el("h3", {}, el("i", { class: `k ${side}` }), `${NAME[side]} words`));
    const grid = el("div", { class: "mini" });
    for (const f of ORDER) {
      const fam = famOf(side, f);
      const x = known[side].by[fam] || { n: 0, examples: [] };
      const p = pct(x.n, known[side].total);
      const lab = el("div", {}, fam);
      const tr = el("div", { class: "track" }, el("div", { class: `bar ${side}`, style: `width:${Math.max(0.4, (78 * p) / fpMax)}%` }),
        el("span", { class: "val" }, el("b", {}, pctStr(p))));
      if (x.examples.length) {
        const ex = () => [el("b", {}, `${fam} · ${fmt(x.n)} words`), el("br"), ...exList(x.examples.slice(0, 6).map((w) => el("span", { class: "w" }, w)))];
        hoverTip(lab, ex); hoverTip(tr, ex);
      }
      grid.append(lab, tr);
    }
    const none = S.fingerprint[side].find((x) => x.family === "No etymology");
    box.append(grid, el("p", { class: "note" }, `${fmt(known[side].total)} words with an etymology; ${fmt(none?.n || 0)} without.`));
    $("#fingerprint").append(box);
  }

  /* 6. hidden cousins */
  let tab = "drifted", shown = 12;
  const card = (c) => {
    const srcLang = c.side, cuzLang = c.side === "nl" ? "en" : "nl";
    return el("div", { class: "card" },
      el("div", { class: "head" }, el("i", { class: `k ${srcLang}` }), link(srcLang, c.word, c.means),
        el("span", { class: "arrow" }, "↔"), el("i", { class: `k ${cuzLang}` }), link(cuzLang, c.cousin)),
      el("div", {}, `${NAME[srcLang]} `, el("span", { class: "w" }, c.word), ` means “${c.means}”; its ${NAME[cuzLang]} cognate `,
        el("span", { class: "w" }, c.cousin), c.cousin_gloss ? el("span", { class: "g" }, ` means “${c.cousin_gloss.replace(/[.…]+$/, "")}…”`) : "", "."),
      el("div", { class: "g" }, `common ancestor: ${c.via_lang} `, el("span", { class: "w" }, c.via)));
  };
  const drawCousins = () => {
    const list = S.cousins[tab];
    $("#cousins").replaceChildren(...list.slice(0, shown).map(card));
    $("#cousins-more").hidden = shown >= list.length;
  };
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
    tab = b.dataset.tab; shown = 12; drawCousins();
  }));
  $("#cousins-more").addEventListener("click", () => { shown += 12; drawCousins(); });
  drawCousins();

  /* 12. uncertainty */
  const U = S.uncertainty;
  const UL = {
    unknown: "marked as of unknown origin",
    uncertain: "uncertain at one or more steps",
    competing: "given more than one possible history (only the first is shown)",
    no_data: "without any etymology on Wiktionary",
  };
  for (const k of ["unknown", "uncertain", "competing", "no_data"]) {
    const cell = (side) => el("td", { class: "num" },
      el("b", {}, pctStr(pct(U[side][k].n, U[side].total))), el("span", { class: "dead" }, ` (${fmt(U[side][k].n)})`), el("br"),
      el("span", { class: "dead" }, ...exList(U[side][k].examples.slice(0, 4).map((w) => link(side, w)))));
    $("#unc tbody").append(el("tr", {}, el("td", {}, UL[k]), cell("en"), cell("nl")));
  }
}
