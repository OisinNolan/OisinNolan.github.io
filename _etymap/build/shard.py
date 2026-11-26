"""Write the site data under /etymap/data/ from build/out/cog_*.

    python build/shard.py

data/index-en.json, data/index-nl.json   headword lists for autocomplete
data/en/<b>.json, data/nl/<b>.json       {"words", "forms", "also"?, "nodes"}, b = crc32(word) % W_BUCKETS (3 hex digits)
data/langs.json                          language table (+ names for unmapped codes)
data/meta.json                           {"word_buckets": W_BUCKETS, "version": 2}
"""
import gzip
import re
import shutil
import zlib
from collections import defaultdict

import orjson

from common import LANGS, OUT, ROOT, SITE

DATA = SITE / "data"
W_BUCKETS = 2048       # word buckets per language (HF serves files uncompressed, so keep them small)


def bucket(s: str, n: int = W_BUCKETS) -> str:
    return f"{zlib.crc32(s.encode()) % n:03x}"


def short_gloss(g: str) -> str:
    g = re.split(r"[;:]|\s\(", g, maxsplit=1)[0].strip().rstrip(",.")
    return g if len(g) <= 44 else g[:42].rsplit(" ", 1)[0] + "…"


def slim(n):
    """[display, lang, gloss, title] -> drop title when it equals the display form."""
    disp, lang, gloss, title = n
    out = [disp, lang, short_gloss(gloss)]
    if title and title not in (disp, disp.lstrip("*")):
        out.append(title)
    while out and out[-1] == "":
        out.pop()
    return out


def main():
    nodes = orjson.loads((OUT / "cog_nodes.json").read_bytes())
    langnames = orjson.loads((OUT / "langnames.json").read_bytes())

    if DATA.exists():
        shutil.rmtree(DATA)
    total, biggest = 0, (0, "")
    used_langs = set()
    for lang in ("en", "nl"):
        shards = defaultdict(lambda: {"words": {}, "forms": {}, "nodes": {}})
        index = []
        with open(OUT / f"cog_{lang}.jsonl", "rb") as f:
            for line in f:
                r = orjson.loads(line)
                w = r["w"]
                sh = shards[bucket(w)]
                entry = {"d": r["d"], "c": r["c"]}
                if any(r["f"]):
                    entry["f"] = r["f"]
                if r["t"]:
                    entry["t"] = r["t"]
                if r.get("s"):
                    entry["s"] = r["s"]
                sh["words"][w] = entry
                ks = {k for dag in r["d"] for k, _, ps in dag} | \
                     {p[0] for dag in r["d"] for _, _, ps in dag for p in ps} | \
                     {k for c in r["c"] if len(c) > 6 for e in c[6] for k in e[:2]}
                for k in ks:
                    n = nodes.get(k) or [k.split("|", 1)[1], k.split("|", 1)[0], "", k.split("|", 1)[1]]
                    sh["nodes"][k] = slim(n)
                    used_langs.add(n[1])
                index.append(w)
        forms = orjson.loads((OUT / f"cog_forms_{lang}.json").read_bytes())
        words = set(index)
        for form, lemma in forms.items():
            if lemma not in words or form == lemma:
                continue
            if form not in words:
                shards[bucket(form)]["forms"][form] = lemma        # took -> take
            else:
                shards[bucket(form)].setdefault("also", {})[form] = lemma   # kijk: a noun, and a form of kijken

        (DATA / lang).mkdir(parents=True, exist_ok=True)
        for p, sh in shards.items():
            b = orjson.dumps(sh)
            (DATA / lang / f"{p}.json").write_bytes(b)
            gz = len(gzip.compress(b, 6))
            total += gz
            if len(b) > biggest[0]:
                biggest = (len(b), f"{lang}/{p}")
        index.sort(key=lambda w: (w.lower(), w))
        b = orjson.dumps(index)
        (DATA / f"index-{lang}.json").write_bytes(b)
        print(f"{lang}: {len(index):,} words, {len(shards)} shards, index {len(gzip.compress(b)) / 1e3:.0f} kB gz")

    table = dict(LANGS)
    for code in sorted(used_langs):
        if code not in table:
            table[code] = {"code": code, "name": langnames.get(code, code)}
    (DATA / "langs.json").write_bytes(orjson.dumps(table))
    (DATA / "meta.json").write_bytes(orjson.dumps({"word_buckets": W_BUCKETS, "version": 2}))
    shutil.copy(ROOT / "data" / "DATA_LICENSE.md", DATA / "LICENSE.md")
    raw = sum(f.stat().st_size for f in DATA.rglob("*.json"))
    print(f"total {raw / 1e6:.1f} MB raw, {total / 1e6:.1f} MB gzipped; largest shard {biggest[1]} {biggest[0] / 1e3:.0f} kB raw")
    unmapped = sorted(c for c in used_langs if c not in LANGS)
    print(f"{len(unmapped)} languages without map position:", unmapped[:40])


if __name__ == "__main__":
    main()
