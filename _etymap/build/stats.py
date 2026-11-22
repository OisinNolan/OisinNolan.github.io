"""Corpus-level statistics for the stats page.

    python build/stats.py      -> /etymap/stats.json (small; committed with the site)

Counts use each headword's *primary* traced translation, so a word with many translations
is counted once. English-side numbers come from English headwords, Dutch-side numbers from
Dutch headwords.
"""
import re
from collections import Counter, defaultdict, deque

import orjson

from common import LANGS, OUT, SITE

EN_LINE = {"en", "enm", "ang", "sco"}
NL_LINE = {"nl", "dum", "odt"}
NATIVE = {"gmw-pro", "gem-pro", "ine-pro", "frk", "gmw"}

FAMILIES = [
    ("Norse", {"non", "gmq-pro", "da", "sv", "is", "no", "nb", "nn", "fo", "gmq", "non-own", "non-oen"}),
    ("French", {"fro", "xno", "frm", "fr", "pro", "fro-nor", "fro-pic", "oc"}),
    ("Latin", {"la", "itc-pro", "la-lat", "la-med", "la-vul", "la-new", "la-ecc", "la-ren", "la-cla"}),
    ("Greek", {"grc", "gkm", "el", "grk-pro", "grc-koi"}),
    ("German", {"de", "gmh", "goh"}),
    ("Low German & Frisian", {"gml", "nds", "osx", "ofs", "fy", "stq", "frr"}),
    ("Other Romance", {"it", "es", "pt", "ca", "ro", "roa-opt", "osp"}),
]
# Known-wrong cousins caused by homographs that share a Wiktionary page (OE god "god" / gōd "good").
# The proper fix is per-etymology glosses in resolve.py; until then keep them out of the headline list.
KNOWN_BAD_COUSINS = {("god", "goed"), ("god", "goei"), ("berg", "berry"), ("eekhoorn", "con")}
STOP = set("a an the of to in on or and for with by as at from be is are any one some that this it its "
           "used especially often usually etc something someone person thing which who".split())


def family(code, side):
    if code in NATIVE or code.endswith("-pro") and code.startswith(("gem", "gmw", "ine")):
        return "Native Germanic"
    if side == "en" and code in NL_LINE:
        return "Dutch"
    if side == "nl" and code in EN_LINE:
        return "English"
    for name, codes in FAMILIES:
        if code in codes:
            return name
    return "Other"


def lang_of(k):
    return k.split("|", 1)[0]


def tokens(s):
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def main():
    nodes = orjson.loads((OUT / "used_nodes.json").read_bytes())
    pairs = {}
    with open(OUT / "pairs.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            pairs[r.pop("id")] = r
    # commonness proxy: number of languages in the English word's translation tables (commonness.py)
    tr_count = orjson.loads((OUT / "commonness.json").read_bytes())
    # Dutch commonness proxy: how many English entries list the word as a translation
    nl_common = Counter()
    with open(OUT / "en_nl.jsonl", "rb") as f:
        for line in f:
            for t in {t["w"] for t in orjson.loads(line)["tr"]}:
                nl_common[t] += 1
    # primary translation per headword: first traced candidate that is a plain word (not 'rechter-',
    # '(teer)geliefd' or a phrase); multi-word headwords are skipped
    plain = re.compile(r"^[^\s()\[\]/,;.!?-][^\s()\[\]/,;.!?]*[^\s()\[\]/,;.!?-]$|^\w$")
    primary, cands = {}, {}
    for side in ("en", "nl"):
        primary[side], cands[side] = {}, {}
        with open(OUT / f"words_{side}.jsonl", "rb") as f:
            for line in f:
                r = orjson.loads(line)
                cands[side][r["w"]] = {c["w"].lower() for c in r["c"]}
                if not plain.match(r["w"]):
                    continue
                pid = next((c["p"] for c in r["c"] if c["p"] and plain.match(c["w"])), None)
                if pid:
                    primary[side][r["w"]] = pid

    disp = lambda k: (nodes.get(k) or [k.split("|", 1)[1]])[0]
    gloss = lambda k: (nodes.get(k) or ["", "", ""])[2] if k in nodes else ""
    common_en = lambda pair: tr_count.get(disp(pair["en"]), 0)

    def dag(pair, side):
        return {k: ps for k, _, ps in pair["A"][side]}

    def chain(pair, side):
        d = dag(pair, side)
        out, k, seen = [], pair[side], set()
        while k and k not in seen:
            seen.add(k)
            ps = d.get(k, [])
            nxt = next((p for p in ps if len(p) < 4 and p[0] in d), None)
            out.append((k, nxt[1] if nxt else None, bool(nxt and nxt[2])))
            k = nxt[0] if nxt else None
        return out   # [(key, edge type to its parent, uncertain)]

    def donor(pair, side):
        """First language outside the word's own line on its main chain, and how it got there."""
        line = EN_LINE if side == "en" else NL_LINE
        ch = chain(pair, side)
        for i, (k, _, _) in enumerate(ch):
            if lang_of(k) not in line:
                return lang_of(k), ch[i - 1][1] if i else None
        return None, None

    def path_types(pair, side, target):
        d = dag(pair, side)
        prev = {pair[side]: None}
        q = deque([pair[side]])
        while q:
            k = q.popleft()
            if k == target:
                types = []
                while prev[k]:
                    k, t = prev[k]
                    types.append(t)
                return types
            for p in d.get(k, []):
                if len(p) >= 4 or p[0] in prev:
                    continue
                prev[p[0]] = (k, p[1])
                q.append(p[0])
        return []

    def example(words, n=6):
        """Best-known examples first; skip proper nouns (Brussels, Wikipedia) unless nothing else is left."""
        ranked = sorted(words, key=lambda x: -x[0])
        lower = [x for x in ranked if all(str(v)[:1].islower() or not str(v)[:1].isalpha() for v in x[1:3] if isinstance(v, str))]
        return (lower + [x for x in ranked if x not in lower])[:n] if len(lower) < n else lower[:n]

    stats = {"counts": {"en_words": len(primary["en"]), "nl_words": len(primary["nl"]), "pairs": len(pairs)}}

    # ---- 1. what kind of pair? (English headwords, primary translation) ----
    kinds = Counter()
    kind_ex = defaultdict(list)
    for w, pid in primary["en"].items():
        p = pairs[pid]
        no_data = [s for s in ("en", "nl") if len(p["A"][s]) <= 1]
        if p["flags"].get("loan"):
            kind = "loan_en" if p["flags"]["loan"] == "en<nl" else "loan_nl"
        elif p["lca"]:
            bor = any(t in ("bor", "cal") for s in ("en", "nl") for t in path_types(p, s, p["lca"][0]))
            kind = "cognate_routes" if bor else "cognate"
        elif no_data:
            kind = "no_data"
        elif any(any(lang_of(b[-1]) in ("en", "nl") for b in g["branches"]) for g in p["ghosts"] if not g.get("loan")):
            kind = "cousin"
        elif any(not g.get("loan") for g in p["ghosts"]):
            kind = "dead_cousin"
        else:
            kind = "unrelated"
        kinds[kind] += 1
        kind_ex[kind].append((common_en(p), w, disp(p["nl"])))
    stats["kinds"] = [{"kind": k, "n": kinds[k], "examples": [[e, d] for _, e, d in example(kind_ex[k])]}
                      for k in ("cognate", "cognate_routes", "loan_en", "loan_nl", "cousin", "dead_cousin", "unrelated", "no_data")]

    # ---- 2. where do cognates meet? (stage of the lowest common ancestor) ----
    lca_lang = Counter()
    lca_ex = defaultdict(list)
    for w, pid in primary["en"].items():
        p = pairs[pid]
        if p["lca"] and not p["flags"].get("loan"):
            c = lang_of(p["lca"][0])
            lca_lang[c] += 1
            lca_ex[c].append((common_en(p), w, disp(p["nl"]), disp(p["lca"][0])))
    stats["lca"] = [{"code": c, "name": LANGS.get(c, {}).get("name", c), "from": LANGS.get(c, {}).get("from"),
                     "to": LANGS.get(c, {}).get("to"), "n": n,
                     "examples": [[e, d, a] for _, e, d, a in example(lca_ex[c], 5)]}
                    for c, n in lca_lang.most_common()]

    # ---- 4. loanword fingerprints ----
    fp = {}
    for side in ("en", "nl"):
        cnt = Counter()
        ex = defaultdict(list)
        for w, pid in primary[side].items():
            p = pairs[pid]
            code, _ = donor(p, side)
            fam = "No etymology" if code is None else family(code, side)
            cnt[fam] += 1
            ex[fam].append((common_en(p), w))
        fp[side] = [{"family": f, "n": n, "examples": [w for _, w in example(ex[f], 6)]} for f, n in cnt.most_common()]
    stats["fingerprint"] = fp

    # ---- 3. native Dutch word vs borrowed English word: replacement, or a new word built in Dutch? ----
    # "replaced": the Dutch word was inherited intact from Proto-West Germanic or older, so English's ancestor
    #             had the word too and English later took a loanword instead (ei / egg, hemel / sky).
    # "built":    the Dutch word was formed inside Dutch or Middle Dutch from native parts (a compound or
    #             derivation), while English borrowed (woordenboek / dictionary). Nothing was replaced.
    groups = {"replaced": (Counter(), defaultdict(list)), "built": (Counter(), defaultdict(list))}
    with_cognate = {"replaced": 0, "survives": 0, "died": 0}
    for w, pid in primary["en"].items():
        p = pairs[pid]
        nl_code, _ = donor(p, "nl")
        en_code, _ = donor(p, "en")
        if nl_code is None or family(nl_code, "nl") != "Native Germanic":
            continue
        if en_code is None or family(en_code, "en") == "Native Germanic":
            continue
        main = chain(p, "nl")
        built = any(lang_of(k) in NL_LINE and lang_of(main[i + 1][0]) in NL_LINE and t in ("cmp", "der")
                    for i, (k, t, _) in enumerate(main[:-1]))
        kind = "built" if built else "replaced"
        fam = family(en_code, "en")
        cnt, ex = groups[kind]
        cnt[fam] += 1
        cousin, dead = None, None
        if kind == "replaced":
            g = next((g for g in p["ghosts"] if g["side"] == "nl" and not g.get("loan")), None)
            keys = [k for k, _, _ in main]
            if g and g["from"] in keys and not any(t == "cmp" for _, t, _ in main[:keys.index(g["from"])]):
                leaf = next((b[-1] for b in g["branches"] if lang_of(b[-1]) == "en"), None)
                cousin = disp(leaf) if leaf else disp(g["branches"][0][-1])
                dead = leaf is None
                if cousin[:1].isupper() or (disp(p["nl"]).lower(), cousin.lower()) in KNOWN_BAD_COUSINS:
                    cousin, dead = None, None
            if cousin:
                with_cognate["replaced"] += 1
                with_cognate["died" if dead else "survives"] += 1
        # examples: prefer ones where the English cognate was found
        ex[fam].append((common_en(p) + (1000 if cousin else 0), w, disp(p["nl"]), cousin, dead))
    stats["replaced"] = {}
    for kind, (cnt, ex) in groups.items():
        stats["replaced"][kind] = {"total": sum(cnt.values()), "by": [
            {"family": f, "n": n, "examples": [[e, d, c, dead] for _, e, d, c, dead in example(ex[f], 8)]}
            for f, n in cnt.most_common()]}
    stats["replaced"]["replaced"]["cognate"] = with_cognate

    # ---- 6. hidden cousins ----
    seen, cousins = set(), []
    for side in ("en", "nl"):
        for w, pid in primary[side].items():
            p = pairs[pid]
            if p["lca"] or p["flags"].get("loan"):
                continue
            for g in p["ghosts"]:
                if g.get("loan"):
                    continue   # borrowed from the word itself (vrolijk -> frolic): not a cognate
                target = "en" if g["side"] == "nl" else "nl"
                main = chain(p, g["side"])
                keys = [k for k, _, _ in main]
                if g["from"] not in keys or any(t == "cmp" for _, t, _ in main[:keys.index(g["from"])]):
                    continue
                leaf = next((b[-1] for b in g["branches"] if lang_of(b[-1]) == target), None)
                if not leaf:
                    continue
                src = p[g["side"]]          # the word whose ancestor also gave `leaf`
                other = p[target]           # its translation in the other language
                key = (src, leaf)
                if key in seen or disp(leaf).lower() == disp(other).lower():
                    continue
                if (disp(src).lower(), disp(leaf).lower()) in KNOWN_BAD_COUSINS:
                    continue
                if disp(leaf).lower() in cands[g["side"]].get(disp(src), set()):
                    continue   # the "cousin" is already a translation of the word
                seen.add(key)
                # meaning drift: does the cousin's gloss share content words with the source word's meaning?
                src_meaning = tokens(gloss(src)) | tokens(disp(other) if g["side"] == "nl" else "")
                if g["side"] == "en":
                    src_meaning |= tokens(disp(src))
                drift = not (tokens(gloss(leaf)) & src_meaning)
                en_word = disp(other) if g["side"] == "nl" else disp(src)
                nl_word = disp(src) if g["side"] == "nl" else disp(leaf)
                cg = gloss(leaf) or ""
                if disp(leaf)[:1].isupper() or "given name" in cg or "surname" in cg or nl_common[nl_word] < 3:
                    continue   # proper names and rare/dialect Dutch words make poor headline examples
                import math
                common = math.log1p(min(tr_count.get(en_word, 0), 800)) + 1.5 * math.log1p(nl_common[nl_word])
                cousins.append({"side": g["side"], "word": disp(src), "means": disp(other), "cousin": disp(leaf),
                                "cousin_gloss": (gloss(leaf) or "")[:60], "via": disp(g["from"]),
                                "via_lang": LANGS.get(lang_of(g["from"]), {}).get("name", lang_of(g["from"])),
                                "drift": drift, "score": common})
    # one entry per cousin word: keep its best-known source
    best = {}
    for c in cousins:
        k = (c["side"], c["cousin"].lower())
        if k not in best or c["score"] > best[k]["score"]:
            best[k] = c
    cousins = sorted(best.values(), key=lambda c: -c["score"])
    stats["cousins"] = {"total": len(cousins),
                        "drifted": [c for c in cousins if c["drift"]][:30],
                        "same": [c for c in cousins if not c["drift"]][:30]}

    # ---- 12. how sure are we? ----
    unc = {}
    for side in ("en", "nl"):
        c = Counter()
        ex = defaultdict(list)
        total = 0
        for w, pid in primary[side].items():
            p = pairs[pid]
            total += 1
            f = set(p["flags"].get(side, []))
            ch = chain(p, side)
            if len(p["A"][side]) <= 1:
                c["no_data"] += 1
                ex["no_data"].append((common_en(p), w))
            if "unknown" in f:
                c["unknown"] += 1
                ex["unknown"].append((common_en(p), w))
            if "uncertain" in f or any(u for _, _, u in ch):
                c["uncertain"] += 1
                ex["uncertain"].append((common_en(p), w))
            if "alt_chain" in f:
                c["competing"] += 1
                ex["competing"].append((common_en(p), w))
        unc[side] = {"total": total, **{k: {"n": c[k], "examples": [w for _, w in example(ex[k], 8)]}
                                         for k in ("unknown", "uncertain", "competing", "no_data")}}
    stats["uncertainty"] = unc

    (SITE / "stats.json").write_bytes(orjson.dumps(stats))
    print(f"wrote {SITE / 'stats.json'} ({(SITE / 'stats.json').stat().st_size / 1e3:.0f} kB)")
    print(orjson.dumps({k: stats[k] for k in ("counts",)}).decode())
    for k in stats["kinds"]:
        print(f"  {k['kind']:15s} {k['n']:6d}  {k['examples'][:4]}")
    print("LCA:", [(x["name"], x["n"]) for x in stats["lca"][:10]])
    for s in ("en", "nl"):
        print(f"fingerprint {s}:", [(x["family"], x["n"]) for x in stats["fingerprint"][s]])
    for kind in ("replaced", "built"):
        R = stats["replaced"][kind]
        print(f"{kind}: {R['total']}", [(x["family"], x["n"], x["examples"][:4]) for x in R["by"][:4]])
    print("  cognate found:", stats["replaced"]["replaced"]["cognate"])
    print("cousins:", stats["cousins"]["total"])
    for c in stats["cousins"]["drifted"][:12]:
        print("   drift", c["side"], c["word"], "=", c["means"], "| cousin", c["cousin"], "-", c["cousin_gloss"][:40])
    for c in stats["cousins"]["same"][:8]:
        print("   same ", c["side"], c["word"], "=", c["means"], "| cousin", c["cousin"], "-", c["cousin_gloss"][:40])
    for s in ("en", "nl"):
        u = stats["uncertainty"][s]
        print(f"uncertainty {s}: total {u['total']}", {k: (u[k]["n"], u[k]["examples"][:4]) for k in ("unknown", "uncertain", "competing", "no_data")})


if __name__ == "__main__":
    main()
