"""Pick interesting Dutch–English relatives for the homepage.

    python build/highlights.py   -> /etymap/highlights.json (small; committed with the site)

Every (Dutch word, English relative) pair from cognates.py is scored on:
  familiarity     both words well known (required)
  drift           the meanings have moved apart (tuin "garden" ~ town)
  recognisable    how alike they sound (sound_similarity in cognates.py)
  elaborateness   languages on the route, borrowings, distance on the map, age of the shared ancestor
and the best go into three categories: drifted apart, distant cousins (PIE), and round trips (words whose route
starts in Europe, leaves it and comes back).
"""
import math
import re
from collections import Counter, deque

import orjson

from cognates import sound_similarity
from common import LANGS, OUT, SITE

PER_CATEGORY = 24
# pairs that score well but are wrong (homographs) or dull; add to this when one shows up on the homepage
BLOCK = {("kerel", "gray"), ("goed", "god"), ("hond", "cynanthropy"), ("als", "also"), ("breken", "broken"),
         ("dat", "sic"), ("duur", "fomite"), ("over", "sore"), ("haak", "hash"),
         ("lopen", "amok"), ("amok", "lope"), ("kameel", "camel"), ("vink", "fink"), ("koraal", "hora"),
         ("bobo", "bobbin"), ("bobo", "booby"), ("romein", "rum"), ("doks", "juke")}
STOP = set("a an the of to in on or and for with by as at from be is are any one some that this it its used especially "
           "often usually etc something someone person thing which who".split())


def lang_of(k):
    return k.split("|", 1)[0]


def dag_map(packed):
    return {k: [list(p) for p in ps] for k, _, ps in packed}


def path_to(m, start, target):
    """Breadth-first up the parents: [start … target] or None."""
    prev = {start: None}
    q = deque([start])
    while q:
        k = q.popleft()
        if k == target:
            out = []
            while k is not None:
                out.append(k)
                k = prev[k]
            return out[::-1]
        for p in m.get(k, ()):
            if len(p) > 3 or p[0] in prev:
                continue
            prev[p[0]] = k
            q.append(p[0])
    return None


def edge_types(m, path):
    out = []
    for c, p in zip(path, path[1:]):
        out.append(next((e[1] for e in m.get(c, ()) if e[0] == p), "inh"))
    return out


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def in_europe(code):
    L = LANGS.get(code)
    if not L or L.get("lat") is None or code.endswith("-pro"):
        return None   # unplaced, or a reconstructed language with a vague homeland
    return L["lat"] >= 35 and L["lon"] <= 42


def round_trip(branch):
    """branch = [lca … word]. The outside-Europe languages if the route starts in Europe, leaves it and
    comes back (Greek -> Arabic -> Latin), else None."""
    seq = []
    for k in branch:
        e = in_europe(lang_of(k))
        if e is not None and (not seq or seq[-1][0] != e):
            seq.append((e, lang_of(k)))
        elif e is False and seq and seq[-1][0] is False:
            seq[-1] = (False, seq[-1][1] + "," + lang_of(k))
    for i in range(1, len(seq) - 1):
        if seq[i][0] is False and seq[i - 1][0] and any(x[0] for x in seq[i + 1:]):
            return seq[i][1].split(",")
    return None


def tokens(s):
    return {t for t in re.findall(r"[a-z]+", (s or "").lower()) if t not in STOP and len(t) > 2}


def short(g):
    g = re.split(r"[;:(]", g or "", maxsplit=1)[0].strip().rstrip(".,")
    return g if len(g) <= 40 else g[:38].rsplit(" ", 1)[0] + "…"


def main():
    nodes = orjson.loads((OUT / "cog_nodes.json").read_bytes())
    tr_count = orjson.loads((OUT / "commonness.json").read_bytes())
    nl_common = Counter()
    with open(OUT / "en_nl.jsonl", "rb") as f:
        for line in f:
            for t in {t["w"] for t in orjson.loads(line)["tr"]}:
                nl_common[t] += 1
    en = {}
    with open(OUT / "cog_en.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            en[r["w"]] = r
    fam_n = lambda n: min(1.0, math.log1p(n) / math.log1p(40))      # Dutch: English entries translating to it
    fam_e = lambda n: min(1.0, math.log1p(n) / math.log1p(150))     # English: languages in its translation table

    rows = []
    with open(OUT / "cog_nl.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            w = r["w"]
            fw = fam_n(nl_common.get(w, 0))
            if nl_common.get(w, 0) < 1 or not r["c"]:   # in at least one English translation table
                continue
            A = dag_map(r["d"][0])
            wgloss = (nodes.get(r["k"]) or ["", "", ""])[2]
            for c in r["c"]:
                cw, ce, wi, lca, dist, same = c[:6]
                if (w, cw) in BLOCK or cw.lower() == w.lower():
                    continue
                o = en.get(cw)
                if not o:
                    continue
                fc = fam_e(tr_count.get(cw, 0))
                if fc < 0.35:
                    continue
                A = dag_map(r["d"][wi] if wi < len(r["d"]) else r["d"][0])
                B = dag_map(o["d"][ce] if ce < len(o["d"]) else o["d"][0])
                for child, par, t in (c[6] if len(c) > 6 else []):
                    B.setdefault(child, []).insert(0, [par, t, 0])
                    B.setdefault(par, [])
                pa, pb = path_to(A, r["k"], lca), path_to(B, o["k"], lca)
                if not pa or not pb:
                    continue
                ta, tb = edge_types(A, pa), edge_types(B, pb)
                types = ta + tb
                if "cmp" in types:
                    continue   # compounds make for muddled one-liners
                trip = round_trip(pa[::-1]) or round_trip(pb[::-1])
                # the round-trip category is rarer, so it gets looser familiarity limits than the others
                if not trip and (fw < 0.45 or fc < 0.55):
                    continue
                if not trip and (len(r["d"]) > 1 or len(o["d"]) > 1):
                    continue   # homograph pages (bij "at" / bij "bee"): the gloss may be the other word's
                pie_nodes = [sum(lang_of(k) == "ine-pro" for k in p) for p in (pa, pb)]
                if lang_of(lca) == "ine-pro" and (max(pie_nodes) > 1 or len(pa) + len(pb) > 13):
                    continue   # long chains of derived PIE roots are where the junk lives (scherp ~ mile)
                unc = any(len(e) > 2 and e[2] for m, p in ((A, pa), (B, pb)) for k in p for e in m.get(k, ()))
                keys = list(dict.fromkeys(pa + pb))
                langs = list(dict.fromkeys(lang_of(k) for k in keys))
                placed = [LANGS[l] for l in langs if l in LANGS and LANGS[l].get("lat") is not None]
                spread = max((km(a, b) for a in placed for b in placed), default=0)
                L = LANGS.get(lang_of(lca), {})
                age = 2026 - (L.get("from") if L.get("from") is not None else 1500)
                cgloss = (nodes.get(o["k"]) or ["", "", ""])[2]
                drift = not same and not (tokens(wgloss) & (tokens(cgloss) | {cw.lower()}))
                sim = sound_similarity("nl", w, "en", cw)
                loan = lca in (r["k"], o["k"])
                fam = math.sqrt(fw * fc)
                elaborate = (min(len(langs), 10) / 10 + min(spread, 4000) / 4000 + min(types.count("bor"), 4) / 4) / 3
                rows.append({
                    "w": w, "c": cw, "fam": fam, "fc": fc,
                    "strict": fw >= 0.45 and fc >= 0.55 and len(r["d"]) == 1 and len(o["d"]) == 1, "sim": sim, "drift": drift, "loan": loan, "unc": unc,
                    "pie": lang_of(lca) == "ine-pro", "germanic": lang_of(lca) in ("gmw-pro", "gem-pro"),
                    "elab": elaborate, "age": age, "langs": len(langs), "km": round(spread), "bor": types.count("bor"),
                    "lca": lca, "via": (nodes.get(lca) or [lca.split("|", 1)[1]])[0],
                    "via_lang": LANGS.get(lang_of(lca), {}).get("name", lang_of(lca)),
                    "wg": short(wgloss), "cg": short(cgloss),
                    "route": [l for l in langs if l not in ("nl", "en")],
                    "trip": [LANGS.get(x, {}).get("name", x) for x in dict.fromkeys(trip or [])],
                })

    def pick(cands, score, n=PER_CATEGORY):
        out, seen_w, seen_c = [], set(), set()
        for x in sorted(cands, key=score, reverse=True):
            if x["w"] in seen_w or x["c"] in seen_c:
                continue   # one card per word on each side
            seen_w.add(x["w"])
            seen_c.add(x["c"])
            out.append(x)
            if len(out) >= n:
                break
        return out

    ok = [x for x in rows if not x["unc"] or x["trip"]]   # trips: loanword routes are often marked uncertain

    def pick_trips(cands):
        # at most three trips through the same outside language (Afrikaans gives many: trek, veld, aardvark)
        out, per = [], Counter()
        for x in pick(cands, lambda x: x["fam"] + 0.3 * x["elab"], n=200):
            key = "+".join(x["trip"])
            if per[key] >= 3:
                continue
            per[key] += 1
            out.append(x)
        return out[:PER_CATEGORY]
    cats = {
        # same root, recognisable, but the meaning moved: tuin ~ town, tijd ~ tide
        "drifted": pick([x for x in ok if x["strict"] and x["drift"] and x["germanic"] and x["sim"] >= 0.5 and not x["loan"]],
                        lambda x: x["fam"] + 0.5 * x["sim"]),
        # meet only in Proto-Indo-European, and don't look alike: hond ~ canine, tand ~ dentist
        "distant": pick([x for x in ok if x["strict"] and x["pie"] and x["bor"] >= 1 and x["fc"] >= 0.68],
                        lambda x: x["fam"] + 0.4 * (1 - x["sim"]) + 0.3 * x["elab"]),
        # starts in Europe, leaves it and comes back: tien ~ dinar (Latin -> Aramaic -> Arabic -> English)
        "journeys": pick_trips([x for x in ok if x["trip"]]),
    }
    keep = ("w", "c", "wg", "cg", "via", "via_lang", "langs", "km", "route", "drift", "trip")
    out = {k: [{f: x[f] for f in keep} for x in v] for k, v in cats.items()}
    (SITE / "highlights.json").write_bytes(orjson.dumps(out))
    print(f"{len(rows):,} pairs scored")
    for k, v in out.items():
        print(f"\n{k}:")
        for x in v[:14]:
            print(f"   {x['w']:<12} ~ {x['c']:<14} {'/'.join(x['trip'])} via {x['via_lang']} {x['via']}  ({x['langs']} langs, {x['km']} km) "
                  f"'{x['wg']}' / '{x['cg'][:30]}'")


if __name__ == "__main__":
    main()
