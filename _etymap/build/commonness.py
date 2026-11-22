"""Commonness proxy for English words: how many languages their Wiktionary translation tables cover.

    python build/commonness.py   -> build/out/commonness.json  {english word: n languages}

Everyday words ('take', 'water') are translated into hundreds of languages; rare words into few.
"""
import gzip

import orjson

from common import DUMP, OUT


def main():
    score = {}
    with gzip.open(DUMP, "rb") as f:
        for line in f:
            if b'"lang_code": "en"' not in line and b'"lang_code":"en"' not in line:
                continue
            e = orjson.loads(line)
            if e.get("lang_code") != "en" or not e.get("translations"):
                continue
            langs = {t.get("lang_code") or t.get("code") for t in e["translations"]}
            w = e["word"]
            score[w] = max(score.get(w, 0), len(langs))
    (OUT / "commonness.json").write_bytes(orjson.dumps(score))
    top = sorted(score.items(), key=lambda kv: -kv[1])[:15]
    print(f"{len(score):,} words; top: {top}")
    print({w: score.get(w) for w in ("take", "water", "dog", "abate", "abstract", "bar", "egg", "hound", "tide")})


if __name__ == "__main__":
    main()
