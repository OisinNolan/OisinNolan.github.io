# Etymap build pipeline

Jekyll ignores this folder (leading underscore). The deployed tool lives in `/etymap/`, and the
post stub is `_posts/2026-11-22-Etymap.html`.

## Rebuild

```sh
cd _etymap
python3 -m venv .venv && .venv/bin/pip install orjson pytest pyyaml
cd build
../.venv/bin/python fetch.py      # ~3.1 GB raw Wiktextract dump -> raw/ (resumable)
../.venv/bin/python extract.py    # ~2 min: graph -> build/out/{nodes,edges,en_nl,nl_en}.jsonl
../.venv/bin/python resolve.py    # ~1 min: pairs, LCAs, ghosts -> build/out/pairs.jsonl etc.
../.venv/bin/python shard.py      # writes /etymap/data/
```

Debug one or more words: `../.venv/bin/python resolve.py --words take,egg,hond` (prints chains).
Test locally from the repo root: `python3 -m http.server 8000`, then open http://localhost:8000/etymap/.

## Site data schema (`/etymap/data/`)

- `meta.json`: `{"buckets": N, "word_buckets": M}` (currently 2048 and 256)
- `langs.json`: `code -> {name, from, to, lat, lon, radius_km, certainty, note}` (mapped languages
  from `data/languages.json`, plus name-only entries for other codes)
- `index-en.json`, `index-nl.json`: sorted headword arrays (autocomplete)
- `en/<b>.json`, `nl/<b>.json`: `{"words": {word: [[translation, sense, pos, pairId|null], …]},
  "forms": {inflected form: lemma}}`, where `b = crc32(utf8(word)) % M` as 3 hex digits
- `p/<crc32(pairId) % N as 3 hex digits>.json`: `{"pairs": {pairId: Pair}, "nodes": {key: [form, lang, gloss?, title?]}}`

`pairId = "en|<en key>~nl|<nl key>"`. A node key is `lang|normalised form` (lowercase, NFC, leading `*`
removed, macrons/breves removed outside reconstructions).

```
Pair = {
  "en": key, "nl": key,
  "A": {"en": [[key, depth, [[parentKey, edgeType, uncertain(0/1), affix?(1)], …]], …], "nl": …},
  "lca": [key, …],                      // minimal common ancestors, best first
  "ghosts": [{"from": key, "side": "en"|"nl", "branches": [[key, …], …], "dead": bool}],
  "flags": {"en": [...], "nl": [...], "loan": "en<nl"|"nl<en"?}   // unknown / uncertain / alt_chain
}
```

Edge types: `inh` inherited, `bor` borrowed, `der` derived, `cal` calque, `cmp` part of a compound.

## Notes on the Wiktextract data (checked 2026-11-22, dump of 2026-10-30)

- Field names match the original plan: `etymology_templates[].{name,args,expansion}`,
  `translations[].{lang_code,code,word,sense}`, `etymology_number` (a **string**, e.g. `"1"`).
- Reconstructions are ordinary entries with `lang_code` like `gem-pro`, `word` without `*`, and
  `original_title: "Reconstruction:Proto-Germanic/nemaną"`.
- `descendants` is a **nested tree** (`{lang, lang_code, word, descendants, raw_tags}`); borrowings are
  marked by `raw_tags: ["borrowed"]` (also "learned borrowing", "uncertain", …).
- Newer pages use `etymon`/`ety` templates (`{{etymon|en|:inh|enm:water<id:liquid>}}`); they're used only
  as a fallback when there's no `inh`/`bor`/`der` chain.
- `etymology_text` can begin with a rendered "Etymology tree" block, which is skipped before matching
  template expansions.
- Etymology-only codes (`la-lat`, `la-med`, `la-vul`, `grc-koi`…) appear as source languages; their
  terms live on the parent language's pages, so keys use the parent code (`common.ETY_ONLY_PARENT`).
- Frankish (`frk`) and Anglo-Norman (`xno`) have no entries of their own; they are leaf nodes.
- Wiktextract occasionally leaks citation text into template args; `valid_term`/`valid_lang` filter it.
- Homographs share a node key, which can conflate unrelated words in the descendants index. The ghost
  search therefore prefers edges from the descendants tree of the ancestor it starts from (edge field `r`).
