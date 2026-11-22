"""Stream the Wiktextract dump and write the etymology graph + translation pairs.

    python build/extract.py [--limit N]

Outputs (build/out/):
  nodes.jsonl   one line per (lang, word, etymology) entry in the language set
  edges.jsonl   child -> parent ancestry edges
  en_nl.jsonl   English entry -> Dutch translations (from translation tables)
  nl_en.jsonl   Dutch entry -> English glosses (links in sense glosses)
  langnames.json  code -> name for every code seen
  stats.json
"""
import argparse
import gzip
import re
from collections import Counter

import orjson

from common import (ANCESTRY, COMPOSED, DUMP, EDGE_TYPE, LANGS, OUT, base_lang,
                    clean_term, key, valid_lang, valid_term)

KEEP = set(LANGS) | {"la", "grc"}
UNCERTAIN = re.compile(r"\b(probably|possibly|perhaps|maybe|likely|uncertain|unclear|disputed|conjectur|may be|might)\b", re.I)
# Text that may sit between two chained templates without breaking the chain.
# The gap between two chained templates must end in a connective ("…, from", "via", "<").
CHAIN_OK = re.compile(r"(?:\b(?:from|via|of)|<|\bborrowed from|\binherited from)\s*$", re.I)
CHAIN_BREAK = re.compile(r"\b(alternatively|compare|cognate|cognates|related|akin|see\s+also|equivalent|doublet|ultimately\s+of|whence)\b|\.\s", re.I)


def targs(t):
    return t.get("args") or {}


def parse_etymon(t, lang):
    """{{etymon|en|:inh|enm:water<id:liquid>|...}} -> [(type, lang, term)]"""
    a = targs(t)
    out, op = [], None
    for k in sorted((k for k in a if k.isdigit() and k != "1"), key=int):
        v = re.sub(r"<[^>]*>", "", a[k]).strip()
        if v.startswith(":"):
            op = v[1:]
        elif ":" in v and op is not None:
            l, term = v.split(":", 1)
            if not valid_lang(l) or not valid_term(clean_term(term)):
                continue
            ty = EDGE_TYPE.get(op, "cmp" if op in {"af", "affix", "compound", "blend"} else "der")
            out.append((ty, l, clean_term(term)))
        elif (op in {"af", "affix", "compound"} or op is None) and valid_term(clean_term(v)):
            out.append(("cmp", lang, clean_term(v)))
    return out


def ancestry_steps(e):
    """Return the linear chain [(type, lang, term, gloss, uncertain, chained_to_prev)] plus
    composition parts and flags, using etymology_text to decide where chains break."""
    lang = e["lang_code"]
    text = e.get("etymology_text") or ""
    if text.startswith("Etymology tree"):
        # Drop the rendered tree: it ends with the headword line, e.g. "English water\n".
        tail = f"{e.get('lang', '')} {e.get('word', '')}\n"
        j = text.find(tail)
        text = text[j + len(tail):] if j >= 0 else text.split("\n")[-1]
    steps, parts, flags = [], [], set()
    pos = 0
    prev_end = None
    etymon = None
    for t in e.get("etymology_templates") or []:
        name = t["name"]
        a = targs(t)
        exp = t.get("expansion") or ""
        if name in {"unk", "unknown"}:
            flags.add("unknown")
            continue
        if name in {"unc", "uncertain"}:
            flags.add("uncertain")
            continue
        if name in ANCESTRY:
            if a.get("1") != lang:
                continue
            src, term = a.get("2", ""), clean_term(a.get("3", ""))
            if not valid_lang(src) or not valid_term(term) or term == "-":
                continue
            gloss = a.get("t") or a.get("gloss") or a.get("5") or ""
            i = text.find(exp, pos) if exp else -1
            between = text[prev_end:i] if (i >= 0 and prev_end is not None) else None
            chained = bool(steps) and between is not None and \
                CHAIN_OK.search(between) is not None and not CHAIN_BREAK.search(between)
            before = text[max(0, (prev_end or 0)):i] if i >= 0 else ""
            unc = bool(UNCERTAIN.search(before)) or "uncertain" in flags
            if steps and not chained:
                # A fresh, unconnected mention (an alternative or a "compare"): keep the first
                # chain only, but remember alternatives as flags.
                flags.add("alt_chain")
                break
            steps.append((EDGE_TYPE[name], src, term, gloss, unc, chained))
            if i >= 0:
                pos = i + len(exp)
                prev_end = pos
        elif name in COMPOSED:
            if a.get("1") != lang or steps:
                # composition of an *ancestor*: attach to the last step if contiguous
                if steps and a.get("1") == steps[-1][1]:
                    parts = [(steps[-1][1], clean_term(a[k])) for k in sorted((k for k in a if k.isdigit() and k != "1"), key=int) if clean_term(a[k])]
                break
            vals = [clean_term(a[k]) for k in sorted((k for k in a if k.isdigit() and k != "1"), key=int)]
            vals = [v for v in vals if v and v != "-"]
            if name in {"suffix", "suf"} and len(vals) >= 2:
                vals = vals[:1] + [v if v.startswith("-") else "-" + v for v in vals[1:]]
            elif name in {"prefix", "pre"} and len(vals) >= 2:
                vals = [v if v.endswith("-") else v + "-" for v in vals[:-1]] + vals[-1:]
            parts = []
            for v in vals:
                pl = lang
                if ":" in v and not v.startswith(":"):
                    maybe, rest = v.split(":", 1)
                    if valid_lang(maybe):
                        pl, v = maybe, rest
                if valid_term(v):
                    parts.append((pl, v))
            break
        elif name in {"etymon", "ety"} and a.get("1") == lang and etymon is None:
            etymon = parse_etymon(t, lang)
    if not steps and not parts and etymon:
        lin = [p for p in etymon if p[0] != "cmp"]
        cmp_ = [p for p in etymon if p[0] == "cmp"]
        if lin:
            ty, src, term = lin[0]
            steps.append((ty, src, term, "", False, False))
        elif cmp_:
            parts = [(l, term) for _, l, term in cmp_]
        flags.add("etymon")
    return steps, parts, flags


def first_gloss(e):
    for s in e.get("senses") or []:
        if "form_of" in s or "alt_of" in s:
            continue
        g = s.get("glosses") or s.get("raw_glosses")
        if g:
            return g[-1][:120]
    return ""


def is_form_only(e):
    ss = e.get("senses") or []
    return bool(ss) and all(("form_of" in s or "alt_of" in s) for s in ss)


def form_targets(e):
    out = []
    for s in e.get("senses") or []:
        for f in (s.get("form_of") or []) + (s.get("alt_of") or []):
            if f.get("word"):
                out.append(f["word"])
    return out


def walk_desc(desc, parent_key, emit, langnames, stats, depth=0, root=None):
    for d in desc or []:
        lc, w = d.get("lang_code"), d.get("word")
        if lc and d.get("lang"):
            langnames.setdefault(lc, d["lang"])
        nxt = parent_key
        if lc and w and base_lang(lc) in KEEP and valid_term(clean_term(w)):
            tags = " ".join((d.get("tags") or []) + (d.get("raw_tags") or [])).lower()
            for rt in (d.get("raw_tags") or []):
                stats["desc_raw_tags"][rt] += 1
            ty = "bor" if ("bor" in tags or "learned" in tags or "→" in tags) else \
                 "cal" if "calque" in tags else "inh"
            k = key(base_lang(lc), clean_term(w))
            if parent_key and k != parent_key:
                emit({"c": k, "p": parent_key, "t": ty, "s": "d", "cf": clean_term(w), "cl": lc,
                      "u": "uncertain" in tags, "r": root})
            nxt = k
        walk_desc(d.get("descendants"), nxt, emit, langnames, stats, depth + 1, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    f_nodes = open(OUT / "nodes.jsonl", "wb")
    f_edges = open(OUT / "edges.jsonl", "wb")
    f_ennl = open(OUT / "en_nl.jsonl", "wb")
    f_nlen = open(OUT / "nl_en.jsonl", "wb")
    langnames = {}
    stats = {"lines": 0, "kept": Counter(), "edges": Counter(), "flags": Counter(),
             "src_lang": Counter(), "desc_raw_tags": Counter(), "en_with_nl": 0, "nl_with_en": 0}

    def emit(ed):
        stats["edges"][ed["t"] + "/" + ed["s"]] += 1
        f_edges.write(orjson.dumps(ed) + b"\n")

    with gzip.open(DUMP, "rb") as f:
        for line in f:
            stats["lines"] += 1
            if args.limit and stats["lines"] > args.limit:
                break
            if stats["lines"] % 1_000_000 == 0:
                print(f"{stats['lines']:,}", flush=True)
            e = orjson.loads(line)
            lc = e.get("lang_code")
            if not lc or "word" not in e:
                continue
            if e.get("lang"):
                langnames.setdefault(lc, e["lang"])
            if lc not in KEEP:
                continue
            stats["kept"][lc] += 1
            word = e["word"]
            ety = e.get("etymology_number") or 0
            k = key(lc, word)
            title = e.get("original_title") or word
            form_only = is_form_only(e)
            node = {"k": k, "w": word, "l": lc, "pos": e.get("pos", ""), "g": first_gloss(e),
                    "e": ety, "ti": title}
            if form_only:
                node["fo"] = form_targets(e)[:3]
            f_nodes.write(orjson.dumps(node) + b"\n")

            if not form_only:
                steps, parts, flags = ancestry_steps(e)
                for fl in flags:
                    stats["flags"][fl] += 1
                child, child_ety = k, ety
                for i, (ty, src, term, gloss, unc, chained) in enumerate(steps):
                    src_b = base_lang(src)
                    stats["src_lang"][src] += 1
                    pk = key(src_b, term)
                    if pk == child:
                        continue
                    emit({"c": child, "ce": child_ety, "p": pk, "t": ty, "s": "s" if i == 0 else "w",
                          "u": unc, "pl": src, "pf": term, "pg": gloss[:80], "src": k})
                    child, child_ety = pk, None
                for pl, term in parts:
                    affix = term.startswith("-") or (term.endswith("-") and not pl.endswith("-pro"))
                    emit({"c": child, "ce": child_ety, "p": key(base_lang(pl), term), "t": "cmp",
                          "s": "s" if not steps else "w", "u": False, "pl": pl, "pf": term, "src": k,
                          "ax": affix})
                if flags & {"unknown", "uncertain", "alt_chain"}:
                    emit({"c": k, "ce": ety, "p": None, "t": "flag", "s": "s",
                          "f": sorted(flags & {"unknown", "uncertain", "alt_chain"}), "src": k})
                walk_desc(e.get("descendants"), k, emit, langnames, stats, root=k)

            if lc == "en" and not form_only:
                tr = [{"w": t["word"], "s": t.get("sense", "")[:100], "tg": t.get("tags", [])}
                      for t in e.get("translations") or []
                      if (t.get("lang_code") or t.get("code")) == "nl" and t.get("word")]
                if tr:
                    stats["en_with_nl"] += 1
                    f_ennl.write(orjson.dumps({"k": k, "w": word, "e": ety, "pos": e.get("pos", ""), "tr": tr}) + b"\n")
            elif lc == "nl" and not form_only:
                glosses = []
                for s in e.get("senses") or []:
                    if "form_of" in s or "alt_of" in s:
                        continue
                    g = (s.get("glosses") or [""])[-1]
                    targets = [l[1] if len(l) > 1 else l[0] for l in s.get("links") or []]
                    glosses.append({"g": g[:120], "l": targets[:4]})
                if glosses:
                    stats["nl_with_en"] += 1
                    f_nlen.write(orjson.dumps({"k": k, "w": word, "e": ety, "pos": e.get("pos", ""), "gl": glosses}) + b"\n")

    for fh in (f_nodes, f_edges, f_ennl, f_nlen):
        fh.close()
    (OUT / "langnames.json").write_bytes(orjson.dumps(langnames, option=orjson.OPT_SORT_KEYS))
    stats["kept"] = dict(stats["kept"].most_common())
    stats["edges"] = dict(stats["edges"].most_common())
    stats["flags"] = dict(stats["flags"].most_common())
    stats["src_lang"] = dict(stats["src_lang"].most_common(120))
    stats["desc_raw_tags"] = dict(stats["desc_raw_tags"].most_common(40))
    (OUT / "stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    print(orjson.dumps({k: stats[k] for k in ("lines", "en_with_nl", "nl_with_en", "edges", "flags")}, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
