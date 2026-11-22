"""Write the site data under /etymap/data/ from build/out/.

    python build/shard.py

data/index-en.json, data/index-nl.json   headword lists for autocomplete
data/en/<b>.json, data/nl/<b>.json       {"words", "forms"}, b = crc32(word) % W_BUCKETS (3 hex digits)
data/p/<b>.json                          {"pairs", "nodes"}, b = crc32(pairId) % N_BUCKETS
data/langs.json                          language table (+ names for unmapped codes)
"""
import gzip
import re
import shutil
import zlib
from collections import defaultdict

import orjson

from common import LANGS, OUT, ROOT, SITE

DATA = SITE / "data"
N_BUCKETS = 2048       # pair buckets (HF serves files uncompressed, so keep them small)
W_BUCKETS = 256        # word buckets per language


def bucket(s: str, n: int = N_BUCKETS) -> str:
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
    pairs = {}
    with open(OUT / "pairs.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            pairs[r.pop("id")] = r
    nodes = orjson.loads((OUT / "used_nodes.json").read_bytes())
    langnames = orjson.loads((OUT / "langnames.json").read_bytes())

    if DATA.exists():
        shutil.rmtree(DATA)
    total, biggest = 0, (0, "")
    used_langs = set()
    for lang in ("en", "nl"):
        shards = defaultdict(lambda: {"words": {}, "forms": {}})
        index = []
        with open(OUT / f"words_{lang}.jsonl", "rb") as f:
            for line in f:
                r = orjson.loads(line)
                w = r["w"]
                shards[bucket(w, W_BUCKETS)]["words"][w] = [[c["w"], c["s"], c["pos"], c["p"]] for c in r["c"]]
                index.append(w)
        forms = orjson.loads((OUT / f"forms_{lang}.json").read_bytes())
        words = set(index)
        for form, lemma in forms.items():
            if form not in words and lemma in words:
                shards[bucket(form, W_BUCKETS)]["forms"][form] = lemma

        (DATA / lang).mkdir(parents=True, exist_ok=True)
        for p, sh in shards.items():
            b = orjson.dumps(sh)
            (DATA / lang / f"{p}.json").write_bytes(b)
            gz = len(gzip.compress(b, 6))
            total += gz
            if gz > biggest[0]:
                biggest = (gz, f"{lang}/{p}")
        index.sort(key=lambda w: (w.lower(), w))
        b = orjson.dumps(index)
        (DATA / f"index-{lang}.json").write_bytes(b)
        print(f"{lang}: {len(index):,} words, {len(shards)} shards, index {len(gzip.compress(b)) / 1e3:.0f} kB gz")

    # pairs: one copy each, bucketed by id hash; both directions point here
    buckets = defaultdict(lambda: {"pairs": {}, "nodes": {}})
    for pid, p in pairs.items():
        b = buckets[bucket(pid)]
        b["pairs"][pid] = p
        ks = {k for side in ("en", "nl") for k, _, ps in p["A"][side]} | \
             {pp[0] for side in ("en", "nl") for _, _, ps in p["A"][side] for pp in ps} | \
             {k for g in p["ghosts"] for br in g["branches"] for k in br} | {g["from"] for g in p["ghosts"]}
        for k in ks:
            n = nodes.get(k) or [k.split("|", 1)[1], k.split("|", 1)[0], "", k.split("|", 1)[1]]
            b["nodes"][k] = slim(n)
            used_langs.add(n[1])
    (DATA / "p").mkdir(parents=True, exist_ok=True)
    for name, b in buckets.items():
        raw_b = orjson.dumps(b)
        (DATA / "p" / f"{name}.json").write_bytes(raw_b)
        gz = len(gzip.compress(raw_b, 6))
        total += gz
        if gz > biggest[0]:
            biggest = (gz, f"p/{name}")

    table = dict(LANGS)
    for code in sorted(used_langs):
        if code not in table:
            table[code] = {"code": code, "name": langnames.get(code, code)}
    (DATA / "langs.json").write_bytes(orjson.dumps(table))
    (DATA / "meta.json").write_bytes(orjson.dumps({"buckets": N_BUCKETS, "word_buckets": W_BUCKETS}))
    shutil.copy(ROOT / "data" / "DATA_LICENSE.md", DATA / "LICENSE.md")
    raw = sum(f.stat().st_size for f in DATA.rglob("*.json"))
    print(f"total {raw / 1e6:.1f} MB raw, {total / 1e6:.1f} MB gzipped shards; largest shard {biggest[1]} {biggest[0] / 1e3:.0f} kB gz")
    unmapped = sorted(c for c in used_langs if c not in LANGS)
    print(f"{len(unmapped)} languages without map position:", unmapped[:40])


if __name__ == "__main__":
    main()
