# OisinNolan.github.io

Personal Jekyll site (GitHub Pages). Posts in `_posts/`, layouts in `_layouts/`. Folders starting with `_`
are ignored by Jekyll except the standard ones.

## Etymap (`/etymap/`, build in `_etymap/`)

Dutch ⇄ English cognate explorer: type a word (NL by default, switch to EN), see its relatives in the
other language, close or distant, drawn as a family tree, a map and a timeline.

- **Front end**: `etymap/index.html`, `app.js` (no build step, d3 + topojson vendored in `lib/`),
  `style.css`, `how.html`, `stats.html` + `stats.js` + `stats.json` (the stats page is about
  *translation* pairs and still comes from `resolve.py` / `stats.py`).
- **Data** (`etymap/data/`, ~145 MB, gitignored) is hosted on the Hugging Face Space
  `oisinn/website_files`; `app.js` reads it from there unless the page is on localhost.
  After any rebuild that changes `etymap/data/`, re-upload with `_etymap/hf_upload.py`
  (the user has to run it, since it needs their HF token). Ask before pushing to GitHub.
- **Pipeline** (`_etymap/build/`, run with `_etymap/.venv/bin/python` from that folder):
  `fetch.py` → `extract.py` → `commonness.py` → `resolve.py` (translation pairs, used by stats) →
  `semantic.py` (synonyms/categories) → `cognates.py` → `shard.py` → `stats.py` → `highlights.py`
  (homepage picks → `etymap/highlights.json`, committed like `stats.json`; when a wrong or dull pair shows up
  there, add it to `BLOCK` in `highlights.py`).
  See `_etymap/README.md` for timings and the data schema.
- **Debug a word**: `../.venv/bin/python cognates.py --words hond,wiel` prints its cognates and
  suggestions (it writes `*.debug` files and never overwrites the full build). The graph load takes
  about 1 minute, so batch the words you want to test into one run.
- **Test locally**: `python3 -m http.server 8000` from the repo root, then open
  http://localhost:8000/etymap/. URLs are `#nl/<word>` or `#nl/<word>/<cognate>`.
- No `node` on this machine. Syntax-check JS with
  `osascript -l JavaScript -e 'ObjC.import("Foundation"); new Function($.NSString.stringWithContentsOfFileEncodingError("app.js",4,null).js); "ok"'`.

### How cognates are found (cognates.py)

- Each word has an ancestry DAG per etymology (`Graph.ancestry` in `resolve.py`). An index maps each
  ancestor to the words in each language that descend from it. A word's cognates are the other-language
  words that share any node with its DAG, grouped by the lowest common ancestor and ordered from
  closest to most distant (allocated breadth-first, so distant PIE-level relatives still get slots).
- The browser rebuilds the pair from the two words' DAGs (`makePair` in `app.js`), so the tree always
  matches the list. Keep the Python `G.lca` and the JS `lcaOf` in sync.
- Noise filters (all in `reach()` and `cognates()`):
  - Don't pass through another modern word of the same language (greyhound → hound).
  - A compound only matches a compound, and then both parts must be shared (nachtmerrie ~ nightmare).
  - A suffixed form made in another language counts as a plain step (canine ← Latin canīnus).
  - PIE cmp edges are root + suffix analyses and are never followed.
  - Words below `MIN_COMMON` commonness are dropped unless they're direct loans.
- **Descendants-tree fallback**: some words don't reach the shared ancestor through their own pages
  (wheel → *hweulō, while wiel → *hwehwlą). Relatives found by walking a Germanic ancestor's descendants
  tree are stored with a `graft` (edges joining them to the ancestor). This only works from Germanic
  protos: further back, homographs get conflated (kaas → chess).
- **Latin/Greek forms**: participles like *subiectum* and *subiectus* are separate pages that only
  say "form of *subiciō*". `Graph.formof` (in `resolve.py`) adds the lemma as an extra `der` parent, so
  *onderwerp* (a calque of *subiectum*) meets *subject* (from *subiectus*).
- **Map positions** live in `_etymap/data/languages.json`. A language missing there shows up in "Not on the
  map", and its arcs are bridged to the nearest mapped ancestor (`.arc.bridge`). Add the language to the
  file to place it. Only `shard.py` needs re-running, since it copies the table to `langs.json`.
- **Wiktionary homographs** share a node key (`lang|form`) across etymologies. This causes most wrong
  links, so check a suspicious result by printing the DAGs before changing ranking.
- Words with no cognate get `s` suggestions: synonyms, related terms, broader terms, the translation's
  own relatives, words near the translation, and words in the same topical category, all of which have
  a cognate.
