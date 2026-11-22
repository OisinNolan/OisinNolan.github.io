"""Shared helpers: language set, term normalisation, template classification."""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "raw" / "raw-wiktextract-data.jsonl.gz"
OUT = ROOT / "build" / "out"
LANGS_FILE = ROOT / "data" / "languages.json"
SITE = ROOT.parent / "etymap"

LANGS = {l["code"]: l for l in json.loads(LANGS_FILE.read_text())}

# Etymology-only codes -> the full language whose pages hold the entries.
# (Wiktionary has no "Late Latin" pages; Late Latin terms live on Latin pages.)
ETY_ONLY_PARENT = {
    "la-lat": "la", "la-vul": "la", "la-med": "la", "la-new": "la", "la-cla": "la",
    "la-ecc": "la", "la-ren": "la", "la-early": "la", "LL.": "la", "ML.": "la", "VL.": "la", "NL.": "la",
    "grc-koi": "grc", "grc-att": "grc", "grc-ion": "grc", "grc-dor": "grc", "grc-aeo": "grc", "grc-epi": "grc",
    "frk": "frk", "ang-nor": "ang", "ang-mer": "ang", "ang-wsx": "ang", "ang-kt": "ang",
    "enm-nor": "enm", "en-GB": "en", "en-US": "en", "nl-BE": "nl", "dum-fla": "dum",
    "fro-nor": "fro", "fro-pic": "fro", "fr-CA": "fr", "non-own": "non", "non-oen": "non", "non-ogt": "non",
    "osx-sax": "osx", "gml-sax": "gml", "de-AT": "de", "de-CH": "de", "goh-bav": "goh",
}

INHERITED = {"inh", "inh+", "inh-lite"}
BORROWED = {"bor", "bor+", "lbor", "slbor", "obor", "ubor", "learned borrowing", "borrowed",
            "semi-learned borrowing", "orthographic borrowing", "lbor+", "slbor+"}
DERIVED = {"der", "der+", "uder", "derived", "der-lite"}
CALQUE = {"cal", "calque", "pcal", "partial calque", "sl", "semantic loan"}
COMPOSED = {"af", "affix", "compound", "com", "com+", "suffix", "suf", "prefix", "pre", "confix", "con",
            "surf", "blend", "univerbation", "infix", "circumfix"}
ANCESTRY = INHERITED | BORROWED | DERIVED | CALQUE
EDGE_TYPE = {**{t: "inh" for t in INHERITED}, **{t: "bor" for t in BORROWED},
             **{t: "der" for t in DERIVED}, **{t: "cal" for t in CALQUE}}
RANK = {"inh": 0, "bor": 1, "cal": 2, "der": 3, "cmp": 4}

_STRIP = {"̄", "̆"}  # macron, breve: dropped from page titles outside reconstructions


def base_lang(code: str) -> str:
    """Etymology-only varieties map to the language whose pages hold their terms: explicit table,
    then the generic rule 'xx-yyy -> xx' when xx is a known language (ang-ang, la-eme, nds-nl, es-MX)."""
    if code in ETY_ONLY_PARENT:
        return ETY_ONLY_PARENT[code]
    if code in LANGS or code.endswith("-pro"):
        return code
    head = code.split("-", 1)[0]
    return head if head != code and head in LANGS else code


def is_proto(code: str) -> bool:
    return code.endswith("-pro")


def norm(lang: str, term: str) -> str:
    t = term.strip().lstrip("*").strip()
    t = unicodedata.normalize("NFD", t)
    if not is_proto(lang):
        t = "".join(c for c in t if c not in _STRIP)
    return unicodedata.normalize("NFC", t).lower()


def key(lang: str, term: str) -> str:
    return f"{lang}|{norm(lang, term)}"


def clean_term(term: str) -> str:
    """Drop wiki markup remnants and inline modifiers like 'enm:water<id:liquid>'."""
    term = re.sub(r"<[^<>]*>", "", term)
    term = re.sub(r"<[^<>]*>", "", term)
    term = re.sub(r"<.*$", "", term).replace(">", "")
    term = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", term)
    return term.strip()


LANG_CODE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4}){0,2}$|^[a-z]{2,3}-[A-Z]{2}$")
_REF = re.compile(r"“|”|!!!|https?:|\bOnline\b|\bPress\b|\b(?:19|20)\d\d\b")


def valid_term(term: str) -> bool:
    return 0 < len(term) <= 48 and any(ch.isalpha() for ch in term) and not _REF.search(term)


def valid_lang(code: str) -> bool:
    return bool(LANG_CODE.match(code or ""))
