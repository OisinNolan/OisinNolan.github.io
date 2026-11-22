"""Phase 1 exploration: stream the dump once, save sample entries and survey the format.

    python build/sample.py
writes raw/samples/dump_<lang>_<word>.json and prints a survey.
"""
import gzip
from collections import Counter
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "raw" / "raw-wiktextract-data.jsonl.gz"
OUT = ROOT / "raw" / "samples"

TARGETS = {
    "en": {"water", "take", "deer", "dog", "cookie", "table", "time", "father", "yacht", "nim", "hound", "tide"},
    "nl": {"water", "nemen", "hert", "dier", "hond", "koekje", "tafel", "tijd", "vader", "jacht"},
    "gem-pro": {"nemaną", "deuzą", "hundaz", "tīdiz"},
    "gmw-pro": {"watar", "neman"},
    "la": {"tabula"},
}
ANCESTRY = {"inh", "inh+", "bor", "bor+", "lbor", "slbor", "obor", "der", "der+", "uder",
            "cal", "calque", "pcal", "af", "affix", "compound", "com", "suffix", "prefix", "confix",
            "root", "unc", "unk", "etymon", "ety"}
KEEP_LANGS = {"en", "enm", "ang", "nl", "dum", "odt", "gmw-pro", "gem-pro", "ine-pro"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    lang_count = Counter()
    tmpl_count = Counter()            # template names in KEEP_LANGS entries
    src_codes = Counter()             # arg 2 of inh/bor/der in KEEP_LANGS entries
    has_orig_title = Counter()
    ety_numbers = Counter()
    nl_tr = 0
    en_entries = 0
    redirects = 0
    examples_orig_title = []

    with gzip.open(DUMP, "rb") as f:
        for line in f:
            n += 1
            e = orjson.loads(line)
            if "redirect" in e or "redirects" in e or "lang_code" not in e:
                redirects += 1
                continue
            lc = e["lang_code"]
            lang_count[lc] += 1
            w = e.get("word", "")
            if "original_title" in e:
                has_orig_title[lc] += 1
                if len(examples_orig_title) < 5 and lc.endswith("-pro"):
                    examples_orig_title.append((w, e["original_title"]))
            if lc in KEEP_LANGS:
                ety_numbers[e.get("etymology_number")] += 1
                for t in e.get("etymology_templates") or []:
                    tmpl_count[t["name"]] += 1
                    if t["name"] in {"inh", "bor", "der", "inh+", "bor+", "der+", "lbor"}:
                        src_codes[t.get("args", {}).get("2")] += 1
            if lc == "en":
                en_entries += 1
                if any(t.get("lang_code") == "nl" or t.get("code") == "nl" for t in e.get("translations") or []):
                    nl_tr += 1
            if w in TARGETS.get(lc, ()):
                pos = e.get("pos", "x")
                num = e.get("etymology_number") or 0
                (OUT / f"dump_{lc}_{w}_{pos}_{num}.json").write_bytes(
                    orjson.dumps(e, option=orjson.OPT_INDENT_2))
            if n % 1_000_000 == 0:
                print(f"{n:,} lines…", flush=True)

    print(f"\nlines {n:,}, redirect/other {redirects:,}")
    print("entries per kept language:", {k: lang_count[k] for k in sorted(KEEP_LANGS)})
    print("other donors:", {k: lang_count[k] for k in ["la", "fro", "xno", "frm", "fr", "non", "frk", "ofs", "osx", "gml", "goh", "grc"]})
    print(f"English entries {en_entries:,}; with ≥1 Dutch translation {nl_tr:,}")
    print("entries with original_title, by lang (top 15):", has_orig_title.most_common(15))
    print("original_title examples:", examples_orig_title)
    print("etymology_number distribution (kept langs):", ety_numbers.most_common(6))
    print("template names (kept langs, top 60):", tmpl_count.most_common(60))
    print("source lang codes in inh/bor/der (top 80):", src_codes.most_common(80))


if __name__ == "__main__":
    main()
