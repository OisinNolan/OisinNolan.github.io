"""Semantic neighbours of English and Dutch headwords, for "related in meaning" suggestions.

    python build/semantic.py   -> build/out/semantic.json  (~3 min: streams the whole dump)

{"en": {word: {"syn": [...], "hyp": [...], "rel": [...], "lnk": [...], "cat": [...]}}, "nl": {...}}

  syn  synonyms                      hyp  hypernyms (dog -> animal)
  rel  coordinate and related terms  lnk  words linked from the first glosses (English only)
  cat  topical categories, e.g. "en:Cervids", "nl:Animals" (not grammatical ones)
"""
import gzip
import re

import orjson

from common import DUMP, OUT

LANGS = ("en", "nl")
TOPICAL = re.compile(r"^(en|nl):")
MAX = 12


def words(objs):
    out = []
    for o in objs or []:
        w = (o.get("word") or "").strip()
        if w and " " not in w and len(w) <= 30:
            out.append(w)
    return out


def add(d, field, ws):
    cur = d.setdefault(field, [])
    for w in ws:
        if w not in cur and len(cur) < MAX:
            cur.append(w)


def main():
    out = {l: {} for l in LANGS}
    pat = [f'"lang_code": "{l}"'.encode() for l in LANGS]
    n = 0
    with gzip.open(DUMP, "rb") as f:
        for line in f:
            if not any(p in line for p in pat):
                continue
            e = orjson.loads(line)
            lc = e.get("lang_code")
            if lc not in out or not e.get("word"):
                continue
            senses = [s for s in e.get("senses") or [] if "form_of" not in s and "alt_of" not in s]
            if not senses:
                continue
            w = e["word"]
            d = out[lc].setdefault(w, {})
            for src in [e] + senses:
                add(d, "syn", words(src.get("synonyms")))
                add(d, "hyp", words(src.get("hypernyms")))
                add(d, "rel", words(src.get("coordinate_terms")) + words(src.get("related")))
            cats = [c for src in [e] + senses for c in src.get("categories") or [] if TOPICAL.match(c)]
            add(d, "cat", cats)
            if lc == "en":
                lnk = [l[1].split("#")[0] if len(l) > 1 else l[0] for s in senses[:2] for l in s.get("links") or []]
                add(d, "lnk", [x for x in lnk if x and " " not in x and x != w])
            for k in [k for k, v in d.items() if not v]:
                del d[k]
            if not d:
                del out[lc][w]
            n += 1
            if n % 200_000 == 0:
                print(f"{n:,}", flush=True)
    (OUT / "semantic.json").write_bytes(orjson.dumps(out))
    print({l: len(v) for l, v in out.items()})
    for l, w in (("nl", "fiets"), ("nl", "hond"), ("en", "bicycle"), ("en", "deer")):
        print(l, w, out[l].get(w))


if __name__ == "__main__":
    main()
