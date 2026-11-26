"""Cognate finder: for every English and Dutch headword, the words in the other language that share
an ancestor with it, however distant (hond ~ hound via Proto-Germanic, ~ canine via PIE).

    python build/cognates.py [--words hond,tuin,fiets]   (subset: prints results, writes *.debug)

Each word gets its own ancestry DAG (the same walk resolve.py uses). Two words are related when their
DAGs share a node; the client rebuilds the pair from the two DAGs, so the tree it draws always agrees
with the cognate list. Words with no relative get "related in meaning" suggestions instead
(synonyms, translations, topical categories; see semantic.py).

Writes build/out/cog_en.jsonl, cog_nl.jsonl ({"w", "k", "d": [dag…], "f": [flags…], "c": [cognate…],
"t": [translation…], "s": [suggestion…]}) and cog_nodes.json (display info for every DAG node).
"""
import argparse
import math
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict, deque

import orjson

from common import OUT, key
from resolve import EN_LINE, NL_LINE, Graph, build_candidates, lang_of, order

OTHER = {"en": "nl", "nl": "en"}
LINE = {"en": EN_LINE, "nl": NL_LINE}
TREE_FROM = {"gem-pro", "gmw-pro"}
MAX_ETYS = 4
PER_ANCESTOR = 60      # candidates kept per shared ancestor (most common first)
PER_GROUP = 6          # cognates shown per common ancestor
MAX_COGNATES = 24
MAX_SUGGEST = 8
MIN_COMMON = 3        # commonness floor for cognates (≈ languages in the translation table)
PLAIN = re.compile(r"^[a-zà-ÿĳ][a-zà-ÿĳ'’]*$")   # lowercase single words; no affixes, names or phrases


def pack(dag):
    return [[k, v["d"], [[p, t, int(u)] + ([1] if ax else []) for p, t, u, ax in v["p"]]]
            for k, v in sorted(dag.items(), key=lambda kv: kv[1]["d"])]


# rough spelling-to-sound keys, so that similarity compares sounds rather than spelling conventions
SOUND = {"nl": [("sch", "sk"), ("ij", "ai"), ("ei", "ai"), ("ui", "au"), ("ou", "au"), ("oe", "u"), ("ie", "i"),
                ("ee", "e"), ("aa", "a"), ("oo", "o"), ("uu", "u"), ("ch", "x"), ("g", "x"), ("w", "v"), ("dt", "t")],
         "en": [("gue", "ge"), ("gui", "gi"), ("kn", "n"), ("wr", "r"), ("tch", "c"), ("ch", "c"), ("sh", "s"), ("igh", "ai"), ("ee", "i"), ("ea", "i"), ("oo", "u"),
                ("ou", "au"), ("ow", "au"), ("gh", "x"), ("ck", "k"), ("ph", "f"), ("wh", "v"), ("w", "v"),
                ("th", "d"), ("c", "k"), ("y", "i"), ("qu", "kv")]}


def sound_key(lang, w):
    w = unicodedata.normalize("NFD", w.lower())
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    for a, b in SOUND[lang]:
        w = w.replace(a, b)
    if lang == "en" and len(w) > 3 and w.endswith("e"):
        w = w[:-1]   # silent final e
    return re.sub(r"(.)\1", r"\1", w)


def sound_similarity(l1, w1, l2, w2):
    """0..1: how alike two words sound (blijken ~ blink high, blijken ~ conflict low)."""
    a, b = sound_key(l1, w1), sound_key(l2, w2)
    cons = lambda x: re.sub(r"[aeiou]", "", x)
    return 0.6 * SequenceMatcher(None, a, b).ratio() + 0.4 * SequenceMatcher(None, cons(a), cons(b)).ratio()


def reach(dag, start, block_same_lang=True):
    """Ancestors of start in its DAG -> (depth, needs_compound).

    needs_compound: the ancestor is only reachable through a compound or a form derived in the
    word's own line (seawater -> water, tidy -> tide + -y). A suffixed form made in another language
    counts as a plain step (canine <- Latin canīnus = canis + -īnus). block_same_lang: don't pass
    through another modern word of the same language (greyhound reaches *hundaz only through hound,
    so it isn't a cognate of hond). PIE "compounds" are root + suffix analyses (*ḱwṓ <- *-ō), so
    those edges are never followed."""
    lang = lang_of(start)
    line = LINE[lang]

    def bfs(allow_cmp):
        out = {start: 0}
        q = deque([start])
        while q:
            n = q.popleft()
            # don't pass through another modern word: of the same language (greyhound -> hound), or of
            # the other one (Dutch waterproof -> English waterproof -> water: it's a relative of
            # waterproof, not of water)
            if block_same_lang and n != start and lang_of(n) in ("en", "nl"):
                continue
            ps = dag.get(n, {}).get("p", ())
            content = [p for p in ps if p[1] == "cmp" and not p[3]]
            for p, t, u, ax in ps:
                if ax or p in out or p not in dag or dag[p].get("ax"):
                    continue
                if t == "cmp":
                    if lang_of(n) == "ine-pro":
                        continue
                    derived_abroad = len(content) == 1 and lang_of(n) not in line
                    if not allow_cmp and not derived_abroad:
                        continue
                out[p] = out[n] + 1
                q.append(p)
        return out

    lineal, full = bfs(False), bfs(True)
    return {m: (d, m not in lineal) for m, d in full.items()
            if not block_same_lang or m == start or lang_of(m) != lang}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="")
    args = ap.parse_args()
    debug = [w for w in args.words.split(",") if w]
    sfx = ".debug" if debug else ""

    G = Graph()
    G.load()
    en_c, nl_c = build_candidates()
    G.common = {key("en", w) for w in en_c} | {key("nl", w) for w in nl_c}
    tr_count = orjson.loads((OUT / "commonness.json").read_bytes())
    sem = orjson.loads((OUT / "semantic.json").read_bytes())

    # --- word pools and commonness ---------------------------------------------------------------
    pool = {"en": {}, "nl": {}}   # lang -> {key: display word}
    for k, n in G.nodes.items():
        l = n[1]
        if l in pool and PLAIN.match(n[0]):
            if l == "en" and n[0] not in tr_count and n[0] not in en_c:
                continue   # English: only words with a translation table (≈ words people know)
            pool[l][k] = n[0]
    comm = {"en": {}, "nl": defaultdict(float)}
    for k, w in pool["en"].items():
        comm["en"][k] = tr_count.get(w, 0) + (5 if w in en_c else 0)
    for e, cs in en_c.items():
        for nlw, _, _, src, _ in cs:
            if src == "tr":
                k = key("nl", nlw)
                comm["nl"][k] = max(comm["nl"][k], tr_count.get(e, 0))
    for nlw, cs in nl_c.items():
        k = key("nl", nlw)
        for e, _, _, src, _ in cs:
            if src == "gl":
                comm["nl"][k] = max(comm["nl"][k], 0.6 * tr_count.get(e, 0))
    print({l: len(p) for l, p in pool.items()}, flush=True)

    # --- DAGs and the ancestor index -------------------------------------------------------------
    dags = {"en": {}, "nl": {}}        # key -> [dag, …] (one per distinct etymology)
    index = {"en": defaultdict(list), "nl": defaultdict(list)}   # ancestor -> [(key, ety idx, depth)]
    for l in ("en", "nl"):
        for i, k in enumerate(pool[l]):
            etys = sorted(G.etys.get(k, {0})) or [0]
            seen, out = set(), []
            for e in etys[:MAX_ETYS]:
                dag = G.ancestry(k, e)
                sig = orjson.dumps(pack(dag))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append((dag, e))
            dags[l][k] = out
            for ei, (dag, _) in enumerate(out):
                for m, (d, cmp) in reach(dag, k).items():
                    index[l][m].append((k, ei, d, cmp))
            if i % 20000 == 0:
                print(f"  {l} dags: {i:,}", flush=True)
    for l in ("en", "nl"):
        for m, lst in index[l].items():
            lst.sort(key=lambda x: (x[3], -comm[l].get(x[0], 0), x[2], len(x[0])))
            del lst[PER_ANCESTOR:]
    print("index built", flush=True)

    translations = {"en": {}, "nl": {}}
    for l, cands in (("en", en_c), ("nl", nl_c)):
        for w, cs in cands.items():
            k = key(l, w)
            if k in pool[l]:
                translations[l][k] = [c[0] for c in order(cs) if " " not in c[0].strip()][:4]

    # --- relatives through descendants trees ------------------------------------------------------
    # Some words don't reach the shared ancestor through their own pages: wheel's page goes back to
    # *hweulō, but *hwehwlą (wiel's ancestor) lists Old English hwēol among its descendants. These
    # relatives are found by walking down descendants trees, as the old "hidden relatives" did, and
    # grafted onto the relative's DAG for display.
    tcache = {}

    def consistent(n, b):
        # Homograph guard: the final word's own page names a parent for a step on the branch in a
        # different language (ME kyken "peep" from Middle Dutch vs kyken "kick" from Norse). A
        # different reconstruction in the same language (*hweulō vs *hwehwlą) is fine.
        leaf, steps = b[-1], [n] + b
        for par, child in zip(steps, steps[1:]):
            said = G.page_says.get((leaf, child))
            same_kind = lambda x: lang_of(x) == lang_of(par) or (lang_of(x).endswith("-pro") and lang_of(par).endswith("-pro"))
            if said and par not in said and not any(same_kind(x) for x in said):
                return False
        return True

    def tree_branches(n, o):
        if (n, o) in tcache:
            return tcache[(n, o)]
        passable = LINE[o] | {"gmw-pro"}
        branches = []

        def dfs(k, path, depth, tree):
            if path and lang_of(k) == o:
                branches.append(path)
                return
            if depth > 5:
                return
            kids = dict.fromkeys(c for c, t, st, r in G.children.get(k, ())
                                 if t != "cmp" and lang_of(c) in passable and c not in path and c != n
                                 and (tree is None or r == tree))
            for c in list(kids)[:12]:
                dfs(c, path + [c], depth + 1, tree)

        dfs(n, [], 0, n)
        if not branches:
            dfs(n, [], 0, None)
        out = [b for b in branches if b[-1] in pool[o] and consistent(n, b)]
        tcache[(n, o)] = out
        return out

    def edge_type(child, parent):
        for c, t, st, r in G.children.get(parent, ()):
            if c == child and t != "cmp":
                return t
        return "inh"

    # --- cognates --------------------------------------------------------------------------------
    def cognates(l, k):
        o = OTHER[l]
        best = {}   # (other key, its ety idx) -> (distance, my ety idx, needs two shared lines, graft)
        for wi, (dag, _) in enumerate(dags[l][k]):
            reached = reach(dag, k, block_same_lang=False)
            for m, (dw, wcmp) in reached.items():
                for c, ce, dc, ccmp in index[o].get(m, ()):
                    loan = m in (k, c)
                    if not loan and comm[o].get(c, 0) < MIN_COMMON:
                        continue   # rare words (kynodesme), unless it's a direct loan
                    if ccmp and not wcmp:
                        continue   # compound of a relative (seawater for water): only for compounds
                    d = dw + dc
                    if (c, ce) not in best or d < best[(c, ce)][0]:
                        best[(c, ce)] = (d, wi, ccmp and wcmp, None)
            found = {c for c, _ in best}
            for m, (dw, wcmp) in sorted(reached.items(), key=lambda x: x[1][0]):
                # Germanic ancestors only: further back, descendants trees conflate homographs
                # (Latin caseus -> chess, Celtic *dūnom -> do)
                if m == k or wcmp or lang_of(m) not in TREE_FROM | LINE[l]:
                    continue
                for b in tree_branches(m, o):
                    c = b[-1]
                    if c in found or comm[o].get(c, 0) < MIN_COMMON:
                        continue
                    cdag = dags[o][c][0][0]
                    # the branch must join the relative's own ancestry right below m (wheel's page knows
                    # Old English hwēol); a later join is accepted only when that ancestry stops early,
                    # with no reconstructed ancestor (churl's stops at Middle English churle). Otherwise
                    # the branch probably went through a homograph (wiel -> well).
                    shallow = not any(x.split("|", 1)[0].endswith("-pro") for x in cdag)
                    joins = b[0] in cdag or (shallow and any(x in cdag for x in b[:-1]))
                    if len(cdag) > 1 and not joins:
                        continue
                    found.add(c)
                    # graft the branch onto c's DAG down to the first node c's own page knows
                    steps, graft = [m] + b, []
                    for par, child in zip(steps, steps[1:]):
                        graft.append([child, par, edge_type(child, par)])
                        if child in cdag:
                            break
                    best[(c, 0)] = (dw + len(b), wi, False, graft[::-1])
        rows = []
        for (c, ce), (d, wi, multi, graft) in best.items():
            if graft:
                lca = [graft[-1][1]]
            else:
                lca = G.lca(dags[l][k][wi][0], dags[o][c][ce][0])
                if not lca or (multi and len(lca) < 2):
                    continue   # compound vs compound: both parts must be shared (nachtmerrie ~ nightmare)
            rows.append((0 if len(lca) > 1 else 1, d, -comm[o].get(c, 0), len(pool[o][c]), c, ce, wi, lca[0], graft))
        tr = {t.lower() for t in translations[l].get(k, ())}
        same = lambda r: pool[o][r[4]].lower() in tr
        rows.sort(key=lambda r: r[:4])
        # group order: the word's main etymology first (fish "animal" before fish "gaming counter"), then a
        # group holding a same-meaning cognate (water ~ water, gast ~ guest), then the closest ancestor
        # the main etymology: the one with a same-meaning relative (four "4" -> vier), else the first
        main_wi = min((r[6] for r in rows if same(r)), default=0)
        gkey = {}
        for r in rows:
            g = gkey.setdefault(r[7], [1, 1, r[0], r[1]])
            g[0] = min(g[0], int(r[6] != main_wi))
            g[1] = min(g[1], 0 if same(r) else 1)
        # within a group: same meaning, then best-known words
        rows.sort(key=lambda r: (gkey[r[7]], not same(r), r[2], r[1], r[3]))
        groups, seen = {}, set()          # lca -> rows, closest ancestor first
        for row in rows:
            if row[4] not in seen:
                seen.add(row[4])
                groups.setdefault(row[7], []).append(row)
        # breadth first: two per ancestor before filling groups up, so distant relatives (often the
        # most surprising ones) aren't crowded out by many close ones
        take = {g: 0 for g in groups}
        total = 0
        for cap in (2, PER_GROUP):
            for g, rs in groups.items():
                while take[g] < min(cap, len(rs)) and total < MAX_COGNATES:
                    take[g] += 1
                    total += 1
        out, ccomm = [], {}
        for g, rs in groups.items():
            for _, d, _, _, c, ce, wi, lca, graft in rs[:take[g]]:
                ccomm[pool[o][c]] = comm[o].get(c, 0)
                row = [pool[o][c], ce, wi, lca, d, int(pool[o][c].lower() in tr)]
                if graft:
                    row.append(graft)
                    for child, par, _ in graft:
                        extra_nodes.update((child, par))
                out.append(row)
        if out:
            # the default (first) relative is the one that sounds most like the word (blijken -> blink, not
            # conflict), from the word's main etymology if it has any there; same meaning breaks near-ties
            w = pool[l][k]
            # commonness boost (scherp -> sharp, not scarp): log-scaled, up to +0.15
            known = lambda r: 0.15 * min(1.0, math.log1p(ccomm.get(r[0], 0)) / math.log1p(150))
            best = max(out, key=lambda r: (r[2] == main_wi, sound_similarity(l, w, o, r[0]) + 0.15 * r[5] + known(r)))
            if sound_similarity(l, w, o, best[0]) < 0.5:
                best = out[0]   # nothing sounds alike (a calque: onderwerp ~ subject): keep the ranked order
            first = [r for r in out if r[3] == best[3]]
            first.remove(best)
            out = [best] + first + [r for r in out if r[3] != best[3]]
        return out

    extra_nodes = set()
    words = {"en": {}, "nl": {}}
    for l in ("en", "nl"):
        for i, (k, w) in enumerate(pool[l].items()):
            words[l][k] = {"w": w, "k": k, "c": cognates(l, k)}
            if i % 20000 == 0:
                print(f"  {l} cognates: {i:,}", flush=True)

    # --- suggestions for words without a relative ------------------------------------------------
    by_word = {l: {w: k for k, w in pool[l].items()} for l in pool}
    cat_members = {l: defaultdict(list) for l in pool}
    for l in pool:
        for w, d in sem[l].items():
            k = by_word[l].get(w)
            if k and words[l][k]["c"]:
                for c in d.get("cat", ()):
                    cat_members[l][c].append(k)
        for c, ks in cat_members[l].items():
            ks.sort(key=lambda x: -comm[l].get(x, 0))

    def best_cog(l, k):
        cs = words[l].get(k, {}).get("c")
        return cs[0][0] if cs else None

    def suggestions(l, k, w):
        o = OTHER[l]
        out, seen = [], {k}
        def add(xk, yw, why, via=None):
            if xk in seen or not yw:
                return
            seen.add(xk)
            out.append([pool[l][xk], yw, why] + ([via] if via else []))
        s = sem[l].get(w, {})
        for field in ("syn", "rel", "hyp"):
            for x in s.get(field, ()):
                xk = by_word[l].get(x)
                if xk:
                    add(xk, best_cog(l, xk), field)
        # its translation's relatives back in this language: fiets -> bicycle ~ wiel
        trs = translations[l].get(k, ())
        close = lambda cs: [c for c in cs if lang_of(c[3]) != "ine-pro"]
        for t in trs:
            tk = by_word[o].get(t)
            for c in close(words[o][tk]["c"] if tk else ())[:2]:
                xk = by_word[l].get(c[0])
                if xk:
                    add(xk, t, "tr", t)
        # words around the translation (bicycle -> wheel, pedal) that have a relative here
        for t in trs[:2]:
            ts = sem[o].get(t, {})
            for x in ts.get("syn", [])[:6] + ts.get("lnk", [])[:6] + ts.get("hyp", [])[:3]:
                xk = by_word[o].get(x)
                for c in close(words[o][xk]["c"] if xk else ())[:1]:
                    ck = by_word[l].get(c[0])
                    if ck:
                        add(ck, x, "near", t)
        # same topic
        cats = sorted(s.get("cat", ()), key=lambda c: len(cat_members[l].get(c, ())) or 1e9)
        for c in cats[:4]:
            for xk in cat_members[l].get(c, [])[:3]:
                add(xk, best_cog(l, xk), "cat", c.split(":", 1)[1])
        out.sort(key=lambda x: x[0].lower() == x[1].lower())   # stable: same-spelled loans last
        return out[:MAX_SUGGEST]

    for l in ("en", "nl"):
        for k, r in words[l].items():
            if not r["c"]:
                r["s"] = suggestions(l, k, r["w"])

    # --- write -----------------------------------------------------------------------------------
    used = set()
    for l in ("en", "nl"):
        with open(OUT / f"cog_{l}.jsonl{sfx}", "wb") as f:
            for k, r in words[l].items():
                if debug and r["w"] not in debug:
                    continue
                r["d"] = [pack(dag) for dag, _ in dags[l][k]]
                r["f"] = [sorted(G.flags.get((k, e), ())) for _, e in dags[l][k]]
                r["t"] = translations[l].get(k, [])
                for dag, _ in dags[l][k]:
                    used.update(dag)
                    used.update(extra_nodes)
                    for v in dag.values():
                        used.update(p for p, *_ in v["p"])
                f.write(orjson.dumps(r) + b"\n")
    (OUT / f"cog_nodes.json{sfx}").write_bytes(orjson.dumps({k: G.display(k) for k in used}))
    for l in ("en", "nl"):
        forms = {f: lem for f, lem in G.forms[l].items() if key(l, lem) in pool[l] and f != lem}
        (OUT / f"cog_forms_{l}.json{sfx}").write_bytes(orjson.dumps(forms))

    n_cog = {l: sum(1 for r in words[l].values() if r["c"]) for l in words}
    n_sug = {l: sum(1 for r in words[l].values() if r.get("s")) for l in words}
    print(f"words: {({l: len(v) for l, v in words.items()})}; with cognates {n_cog}; "
          f"suggestions for {n_sug}; nodes {len(used):,}")

    if debug:
        for l in ("en", "nl"):
            for k, r in words[l].items():
                if r["w"] not in debug:
                    continue
                print(f"\n== {l} {r['w']}  tr={r['t']}")
                for c in r["c"]:
                    print(f"   {c[0]:<16} d={c[4]} via {G.display(c[3])[1]} {G.display(c[3])[0]}" + (" (same meaning)" if c[5] else "") + (" [tree]" if len(c) > 6 else ""))
                for s in r.get("s", []):
                    print("   suggest:", s)


if __name__ == "__main__":
    main()
