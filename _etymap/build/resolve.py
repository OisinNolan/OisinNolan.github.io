"""Resolve English<->Dutch word pairs: translations, ancestry DAGs, LCAs, ghost branches.

    python build/resolve.py [--words take,water]   (subset for debugging, prints results)

Writes build/out/words_en.jsonl, words_nl.jsonl (candidates per headword),
pairs.jsonl (one resolved pair per line), forms_en/nl.json and used_nodes.json.
"""
import argparse
import re
from collections import defaultdict, deque

import orjson

from common import LANGS, OUT, RANK, key, norm

EN_LINE = {"en", "enm", "ang", "sco"}
NL_LINE = {"nl", "dum", "odt"}
MODERN = {"en": "en", "nl": "nl"}
MAX_DEPTH = 16
MAX_NODES = 32
MAX_CANDIDATES = 8
MAX_RESOLVED = 4


def ety_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def is_affix(k):
    lang, t = k.split("|", 1)
    t = t.replace("*", "")
    return t.startswith("-") or (t.endswith("-") and not lang.endswith("-pro"))


def lang_of(k):
    return k.split("|", 1)[0]


class Graph:
    def __init__(self):
        self.parents = defaultdict(list)   # child -> [edge]
        self.children = defaultdict(list)  # parent -> [(child, type)]
        self.flags = defaultdict(set)      # (key, ety) -> flags
        self.fallback = {}                 # key -> (display, lang) from edges
        self.nodes = {}                    # key -> [display, lang, gloss, title]
        self.etys = defaultdict(set)       # key -> etymology numbers present
        self.forms = {"en": {}, "nl": {}}  # form -> lemma
        self.strength = {}                 # (child, parent) -> 0 own page, 1 descendants/chain
        self.common = set()                # keys of headwords with translations (≈ common words)

    def load(self):
        print("loading edges…", flush=True)
        with open(OUT / "edges.jsonl", "rb") as f:
            for line in f:
                e = orjson.loads(line)
                if e["t"] == "flag":
                    self.flags[(e["c"], ety_int(e.get("ce")))].update(e["f"])
                    continue
                c, p = e["c"], e["p"]
                if not p or c == p:
                    continue
                self.parents[c].append((p, e["t"], e["s"], ety_int(e.get("ce")), e.get("src"), e.get("u", False), e.get("ax", False)))
                self.children[p].append((c, e["t"], e["s"], e.get("r")))
                st = 0 if e["s"] == "s" else 1
                if st < self.strength.get((c, p), 9):
                    self.strength[(c, p)] = st
                if "pf" in e:
                    self.fallback.setdefault(p, (e["pf"], e.get("pl", lang_of(p))))
                if "cf" in e:
                    self.fallback.setdefault(c, (e["cf"], e.get("cl", lang_of(c))))
        print(f"  {sum(map(len, self.parents.values())):,} edges", flush=True)
        print("loading nodes…", flush=True)
        with open(OUT / "nodes.jsonl", "rb") as f:
            for line in f:
                n = orjson.loads(line)
                k, l = n["k"], n["l"]
                if "fo" in n:
                    if l in self.forms and n["fo"]:
                        self.forms[l].setdefault(n["w"], n["fo"][0])
                    continue
                self.etys[k].add(ety_int(n["e"]) or 0)
                cur = self.nodes.get(k)
                better = cur is None or (not cur[2] and n["g"]) or \
                    (cur[0] != cur[0].lower() and n["w"] == n["w"].lower())
                if better:
                    disp = ("*" + n["w"]) if l.endswith("-pro") and not n["w"].startswith("*") else n["w"]
                    self.nodes[k] = [disp, l, n["g"], n["ti"]]
        print(f"  {len(self.nodes):,} nodes", flush=True)

    # --- ancestry -----------------------------------------------------------------

    def best_parents(self, n, ety=None, root=None):
        es = self.parents.get(n, ())
        # A homograph (several etymologies on one page) reached without knowing which one is meant:
        # trust the chain written on the starting word's own page, e.g. eye -> ME e -> OE ēage,
        # rather than ME e "he" -> OE hē.
        if ety is None and root and root != n and len(self.etys.get(n, ())) != 1:
            hint = [e for e in es if e[4] == root and e[1] != "cmp"]
            if hint:
                best = min(enumerate(hint), key=lambda ie: (RANK.get(ie[1][1], 9), ie[0]))[1]
                return [(best[0], best[1], best[5], False)]
        own = [e for e in es if e[2] == "s" and e[4] == n]
        if ety is not None:
            own_e = [e for e in own if e[3] == ety]
            if own_e:
                own = own_e
        for tier in (own, [e for e in es if e[2] == "d"], [e for e in es if e[2] == "w"]):
            if not tier:
                continue
            lin = [e for e in tier if e[1] != "cmp"]
            if lin:
                best = min(enumerate(lin), key=lambda ie: (RANK.get(ie[1][1], 9), ie[0]))[1]
                return [(best[0], best[1], best[5], False)]
            parts, seen = [], set()
            for e in tier:
                if e[0] not in seen:
                    seen.add(e[0])
                    parts.append((e[0], "cmp", e[5], e[6] or is_affix(e[0])))
            return parts
        return []

    def ancestry(self, start, ety):
        """DAG of ancestors: {key: {"d": depth, "p": [(parent, type, uncertain, affix)]}}"""
        dag = {start: {"d": 0, "p": []}}
        queue = deque([(start, ety, 0)])
        while queue:
            n, e, d = queue.popleft()
            if d >= MAX_DEPTH or len(dag) >= MAX_NODES:
                continue
            ps = self.best_parents(n, e, start)
            ps = [x for x in ps if any(ch.isalpha() for ch in x[0].split("|", 1)[1])]
            dag[n]["p"] = ps
            for p, t, u, ax in ps:
                if not any(ch.isalpha() for ch in p.split("|", 1)[1]):
                    continue
                if ax:
                    dag.setdefault(p, {"d": d + 1, "p": [], "ax": True})
                    continue
                if p in dag:
                    continue
                dag[p] = {"d": d + 1, "p": []}
                queue.append((p, None, d + 1))
        return dag

    @staticmethod
    def ancestors_of(dag, n):
        out, stack = set(), [n]
        while stack:
            for p, *_ in dag.get(stack.pop(), {}).get("p", ()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def lca(self, da, db):
        common = {k for k in da if k in db and not da[k].get("ax") and not db[k].get("ax")}
        if not common:
            return []
        lowest = []
        for c in common:
            if not any(c in self.ancestors_of(da, o) or c in self.ancestors_of(db, o) for o in common if o != c):
                lowest.append(c)
        return sorted(lowest, key=lambda c: da[c]["d"] + db[c]["d"])

    def ghosts(self, dag, other_dag, other_line, exclude):
        """Lowest node on `dag` with descendants in the other language line that aren't on
        the other word's own path. Returns {"from", "branches": [[keys…]], "dead"} or None.
        Branches stop at the first modern-language word (no compounds like nim -> nimber)."""
        modern = MODERN["en" if other_line is EN_LINE else "nl"]
        passable = other_line | {"gmw-pro"}
        for n in sorted(dag, key=lambda k: dag[k]["d"]):
            if n in exclude or dag[n].get("ax") or dag[n]["d"] == 0 or lang_of(n) == "ine-pro":
                continue
            branches = []

            def dfs(k, path, depth, tree):
                if path and lang_of(k) == modern:
                    branches.append(path)
                    return
                kids = list(dict.fromkeys(
                    c for c, t, st, r in self.children.get(k, ())
                    if t != "cmp" and lang_of(c) in passable and c not in path and c != n
                    and (tree is None or r == tree)))
                if not kids or depth > 5:
                    if path and lang_of(path[-1]) in other_line:
                        branches.append(path)
                    return
                for c in kids[:12]:
                    dfs(c, path + [c], depth + 1, tree)

            # Prefer the descendants tree on n's own page (internally consistent); fall back
            # to all edges, where homographs can be conflated.
            dfs(n, [], 0, n)
            if not branches:
                dfs(n, [], 0, None)
            branches = [b for b in branches if not any(k in other_dag for k in b)]
            if branches:
                def rank(b):
                    leaf = b[-1]
                    return (lang_of(leaf) != modern, leaf not in self.common,
                            self.strength.get((leaf, b[-2] if len(b) > 1 else n), 2), len(b),
                            len(leaf))
                branches.sort(key=rank)
                uniq, seen_leaf = [], set()
                for b in branches:
                    if b[-1] not in seen_leaf:
                        seen_leaf.add(b[-1])
                        uniq.append(b)
                uniq = uniq[:3]
                dead = not any(lang_of(b[-1]) == modern for b in uniq)
                return {"from": n, "branches": uniq, "dead": dead}
        return None

    def resolve(self, en_k, en_ety, nl_k, nl_ety):
        best = None
        en_etys = [en_ety] if en_ety is not None else (sorted(self.etys.get(en_k, {0})) or [0])
        nl_etys = [nl_ety] if nl_ety is not None else (sorted(self.etys.get(nl_k, {0})) or [0])
        for ee in en_etys[:4]:
            for ne in nl_etys[:4]:
                da, db = self.ancestry(en_k, ee), self.ancestry(nl_k, ne)
                l = self.lca(da, db)
                score = (bool(l), -(da[l[0]]["d"] + db[l[0]]["d"]) if l else -99, len(da) + len(db))
                if best is None or score > best[0]:
                    best = (score, ee, ne, da, db, l)
        _, ee, ne, da, db, l = best
        exclude_a = set(l) | set().union(*(self.ancestors_of(da, c) for c in l)) if l else set()
        exclude_b = set(l) | set().union(*(self.ancestors_of(db, c) for c in l)) if l else set()
        ghosts = []
        g = self.ghosts(db, da, EN_LINE, exclude_b)
        if g:
            g["side"] = "nl"
            ghosts.append(g)
        g = self.ghosts(da, db, NL_LINE, exclude_a)
        if g:
            g["side"] = "en"
            ghosts.append(g)
        flags = {"en": sorted(self.flags.get((en_k, ee), ())), "nl": sorted(self.flags.get((nl_k, ne), ()))}
        if l:
            if l[0] == nl_k:
                flags["loan"] = "en<nl"
            elif l[0] == en_k:
                flags["loan"] = "nl<en"

        def pack(dag):
            return [[k, v["d"], [[p, t, int(u)] + ([1] if ax else []) for p, t, u, ax in v["p"]]]
                    for k, v in sorted(dag.items(), key=lambda kv: kv[1]["d"])]

        return {"en": en_k, "nl": nl_k, "A": {"en": pack(da), "nl": pack(db)}, "lca": l,
                "ghosts": ghosts, "flags": flags}

    def display(self, k):
        n = self.nodes.get(k)
        if n:
            return n
        disp, l = self.fallback.get(k, (k.split("|", 1)[1], lang_of(k)))
        if l.endswith("-pro"):
            disp = "*" + disp.replace("*", "")
        return [disp, lang_of(k), "", disp.lstrip("*")]


# --- translation candidates ----------------------------------------------------------

PAREN = re.compile(r"\([^)]*\)")
LEAD = re.compile(r"^(to|a|an|the)\s+", re.I)


def gloss_pieces(g):
    g = PAREN.sub("", g)
    return [LEAD.sub("", p.strip()).lower() for p in re.split(r"[;,]", g) if p.strip()]


def build_candidates():
    en = defaultdict(list)   # en word -> [(nl word, sense, en_ety, src)]
    nl = defaultdict(list)   # nl word -> [(en word, sense, nl_ety, src)]
    with open(OUT / "en_nl.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            for t in r["tr"]:
                en[r["w"]].append((t["w"], t["s"], ety_int(r["e"]) or 0, "tr", r["pos"]))
                nl[t["w"]].append((r["w"], t["s"], None, "rtr", r["pos"]))
    with open(OUT / "nl_en.jsonl", "rb") as f:
        for line in f:
            r = orjson.loads(line)
            for gl in r["gl"]:
                pieces = gloss_pieces(gl["g"])
                for tgt in gl["l"]:
                    if tgt.lower() in pieces:
                        nl[r["w"]].append((tgt, gl["g"], ety_int(r["e"]) or 0, "gl", r["pos"]))
                        en[tgt].append((r["w"], gl["g"], None, "rgl", r["pos"]))
    return en, nl


def order(cands):
    """Keep source order but put direct sources (tr / gl) before reversed ones; dedupe."""
    pri = {"tr": 0, "gl": 0, "rtr": 1, "rgl": 1}
    out, seen = [], set()
    for c in sorted(enumerate(cands), key=lambda ic: (pri[ic[1][3]], ic[0])):
        c = c[1]
        if c[0] in seen:
            continue
        seen.add(c[0])
        out.append(c)
    return out[:MAX_CANDIDATES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="")
    args = ap.parse_args()
    G = Graph()
    G.load()
    en_c, nl_c = build_candidates()
    G.common = {key("en", w) for w in en_c} | {key("nl", w) for w in nl_c}
    print(f"candidates: {len(en_c):,} English, {len(nl_c):,} Dutch headwords", flush=True)

    debug = [w for w in args.words.split(",") if w]
    sfx = ".debug" if debug else ""   # debug runs never overwrite the full build output
    pairs = {}
    used = set()

    def pair(en_w, en_ety, nl_w, nl_ety):
        ek, nk = key("en", en_w), key("nl", nl_w)
        pid = f"{ek}~{nk}"
        if pid not in pairs:
            r = G.resolve(ek, en_ety, nk, nl_ety)
            pairs[pid] = r
            for side in ("en", "nl"):
                for k, _, ps in r["A"][side]:
                    used.add(k)
            for g in r["ghosts"]:
                for b in g["branches"]:
                    used.update(b)
        return pid

    def run(direction, cands, fout):
        n = 0
        for w, cs in cands.items():
            if debug and w not in debug:
                continue
            if direction == "en" and key("en", w) not in G.nodes:
                continue
            if direction == "nl" and key("nl", w) not in G.nodes:
                continue
            out = []
            for i, (other, sense, ety, src, pos) in enumerate(order(cs)):
                pid = None
                if i < MAX_RESOLVED and " " not in other.strip():
                    pid = pair(w, ety, other, None) if direction == "en" else pair(other, None, w, ety)
                out.append({"w": other, "s": sense, "pos": pos, "p": pid})
            used.add(key(direction, w))
            fout.write(orjson.dumps({"w": w, "c": out}) + b"\n")
            n += 1
            if n % 20000 == 0:
                print(f"  {direction}: {n:,}", flush=True)
        return n

    with open(OUT / f"words_en.jsonl{sfx}", "wb") as fe, open(OUT / f"words_nl.jsonl{sfx}", "wb") as fn:
        ne = run("en", en_c, fe)
        nn = run("nl", nl_c, fn)
    print(f"words: {ne:,} en, {nn:,} nl; pairs: {len(pairs):,}", flush=True)

    with open(OUT / f"pairs.jsonl{sfx}", "wb") as f:
        for pid, r in pairs.items():
            f.write(orjson.dumps({"id": pid, **r}) + b"\n")

    nodes = {k: G.display(k) for k in used}
    (OUT / f"used_nodes.json{sfx}").write_bytes(orjson.dumps(nodes))
    for l in ("en", "nl"):
        forms = {f: lem for f, lem in G.forms[l].items() if key(l, lem) in used and f != lem}
        (OUT / f"forms_{l}.json{sfx}").write_bytes(orjson.dumps(forms))

    # coverage + language check
    langs_used = defaultdict(int)
    for k in used:
        langs_used[lang_of(k)] += 1
    missing = {l: c for l, c in sorted(langs_used.items(), key=lambda x: -x[1]) if l not in LANGS}
    with_lca = sum(1 for r in pairs.values() if r["lca"])
    with_ghost = sum(1 for r in pairs.values() if r["ghosts"])
    stats = {"words_en": ne, "words_nl": nn, "pairs": len(pairs), "pairs_with_lca": with_lca,
             "pairs_with_ghost": with_ghost, "nodes_used": len(nodes),
             "langs_missing_from_table": dict(list(missing.items())[:60])}
    (OUT / f"resolve_stats.json{sfx}").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    print(orjson.dumps({k: v for k, v in stats.items() if k != "langs_missing_from_table"}).decode())
    print("missing langs (top):", list(missing.items())[:25])

    if debug:
        for pid, r in pairs.items():
            print("\n=====", pid, "LCA:", [G.display(k)[:2] for k in r["lca"]], "flags:", r["flags"])
            for side in ("en", "nl"):
                print(f"  {side}:", " | ".join(f"{G.display(k)[1]} {G.display(k)[0]}" + (f" <-{','.join(p[1] for p in ps)}" if ps else "")
                                            for k, d, ps in r["A"][side]))
            for g in r["ghosts"]:
                print(f"  ghost ({g['side']}) from {G.display(g['from'])[:2]} dead={g['dead']}:",
                      [[G.display(k)[0] + "/" + lang_of(k) for k in b] for b in g["branches"]])


if __name__ == "__main__":
    main()
