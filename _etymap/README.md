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
../.venv/bin/python commonness.py # word commonness (translation-table sizes)
../.venv/bin/python resolve.py    # ~1 min: translation pairs (for the stats page) -> build/out/pairs.jsonl etc.
../.venv/bin/python semantic.py   # ~1 min: synonyms, related terms, topical categories -> build/out/semantic.json
../.venv/bin/python cognates.py   # ~1 min: every word's DAG + cognates + suggestions -> build/out/cog_*
../.venv/bin/python shard.py      # writes /etymap/data/
../.venv/bin/python stats.py      # writes /etymap/stats.json
../.venv/bin/python highlights.py # writes /etymap/highlights.json (homepage picks)
```

Debug one or more words: `../.venv/bin/python cognates.py --words hond,wiel` (cognates, suggestions) or
`resolve.py --words take,egg` (translation pairs).
Test locally from the repo root: `python3 -m http.server 8000`, then open http://localhost:8000/etymap/.

## Site data schema (`/etymap/data/`, version 2)

- `meta.json`: `{"word_buckets": 2048, "version": 2}`
- `langs.json`: `code -> {name, from, to, lat, lon, radius_km, certainty, note}`
- `index-en.json`, `index-nl.json`: sorted headword arrays (autocomplete)
- `en/<b>.json`, `nl/<b>.json`, `b = crc32(utf8(word)) % 2048` as 3 hex digits:

```
{"words": {word: Entry}, "forms": {inflected form: lemma}, "also": {headword that is also a form: lemma},
 "nodes": {key: [form, lang, gloss?, title?]}}
Entry = {
  "d": [Dag, …],          // one per distinct etymology; Dag = [[key, depth, [[parentKey, type, unc, affix?]]], …]
  "f": [[flag…], …],      // per etymology: unknown / uncertain / alt_chain (omitted when empty)
  "t": [word, …],         // translations into the other language (omitted when none)
  "c": [[word, theirEty, myEty, lcaKey, distance, sameMeaning(0/1), graft?], …],
  "s": [[word here, its cognate, why, via?], …]   // only when "c" is empty
}
graft = [[child, parent, type], …]   // lowest edge first; joins the cognate's DAG to lcaKey
```

A node key is `lang|normalised form` (lowercase, NFC, leading `*` removed, macrons/breves removed
outside reconstructions).

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
