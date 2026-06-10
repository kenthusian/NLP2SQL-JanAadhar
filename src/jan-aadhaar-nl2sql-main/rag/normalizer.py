"""
rag/normalizer.py — Phase 2a: Typo normalization using RapidFuzz.

Corrects misspelled district/city names and other controlled vocabulary
before the query reaches the cache or LLM.

Usage:
    from rag.normalizer import normalize_query
    clean = normalize_query("show me records from jaipor district")
    # -> "show me records from Jaipur district"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger

log = get_logger("nl2sql.normalizer")

# ── Rajasthan districts vocabulary (real names from the dataset) ──────────────
_DISTRICTS = [
    "Ajmer", "Alwar", "Balotara", "Banswara", "Baran", "Barmer",
    "Beawar", "Bharatpur", "Bhilwara", "Bikaner", "Bundi",
    "Chittorgarh", "Churu", "Dausa", "Deeg", "Dholpur",
    "Didwana-Kuchaman", "Dungarpur", "Hanumangarh", "Jaipur",
    "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur",
    "Karauli", "Khairthal-Tijara", "Kota", "Kotputli-Behror",
    "Nagaur", "Pali", "Phalodi", "Pratapgarh", "Rajsamand",
    "Sawai Madhopur", "Sikar", "Sirohi", "Sri Ganganagar",
    "Tonk", "Udaipur",
]

# ── General controlled vocabulary ────────────────────────────────────────────
# Maps misspelling → correct form (exact case-insensitive replacement)
_VOCAB_MAP: dict[str, str] = {
    # gender
    "male":    "Male",
    "female":  "Female",
    "males":   "Male",
    "females": "Female",
    # rural/urban
    "rural":   "rural",
    "urban":   "urban",
    # marital
    "married":   "Married",
    "unmarried": "Unmarried",
    "widow":     "Widowed",
    "widowed":   "Widowed",
    "divorced":  "Divorced",
    # caste
    "general": "GEN",
    "gen":     "GEN",
    "obc":     "OBC",
    "sc":      "SC",
    "st":      "ST",
    # education
    "graduate":     "Graduate",
    "graduates":    "Graduate",
    "illiterate":   "Illiterate",
    "literate":     "Literate",
    "postgraduate": "Post Graduate",
    "post graduate":"Post Graduate",
    # occupation
    "farmer":    "Farmer",
    "farmers":   "Farmer",
    "labourer":  "Labourer",
    "laborer":   "Labourer",
    "laborers":  "Labourer",
    "labour":    "Labourer",
    "labor":     "Labourer",
    "student":   "Student",
    "students":  "Student",
    "housewife": "Home Maker",
    "homemaker": "Home Maker",
    "unemployed":"Unemployed",
    # common field typos
    "icome":     "income",
    "incom":     "income",
    "salry":     "salary",
    "lak":       "lakh",
    "laks":      "lakhs",
    "distric":   "district",
    "villag":    "village",
    "beople":    "people",
    "poeple":    "people",
    "peopl":     "people",
    "peple":     "people",
}

# ── Fuzzy district correction ─────────────────────────────────────────────────
# Only imported if rapidfuzz is available
_FUZZY_THRESHOLD = 80   # minimum similarity score (0-100) to accept a correction

try:
    from rapidfuzz import process as _rfprocess, fuzz as _rffuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.warning("rapidfuzz not installed; district typo correction disabled")

import json
_VILLAGES: list[str] = []
def _load_villages() -> None:
    p = Path(__file__).resolve().parents[1] / "data" / "villages.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                _VILLAGES.extend(json.load(f))
        except Exception as e:
            log.error(f"Failed to load villages: {e}")
_load_villages()

_IGNORE_LIST = {"saini", "sharma", "verma", "yadav", "gupta", "jain", "khan", "singh", "kumar"}
def _load_ignore_list() -> None:
    p = Path(__file__).resolve().parents[1] / "data" / "caste_mapping.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                _IGNORE_LIST.update(json.load(f).keys())
        except Exception as e:
            log.error(f"Failed to load caste mappings: {e}")
_load_ignore_list()


_DOMAIN_KEYWORDS = [
    "people", "members", "district", "village", "income", "salary",
    "female", "male", "rural", "urban", "married", "unmarried",
    "widow", "divorced", "general", "graduate", "illiterate",
    "literate", "farmer", "labourer", "student", "homemaker",
    "unemployed", "family", "households", "children", "adults",
    "count", "list", "show", "average", "total", "minimum", "maximum"
]

def _fuzzy_correct_word(word: str) -> str:
    """
    If *word* looks like a district name or village name, try fuzzy-matching it to the
    canonical list. Returns the corrected name if confidence >= threshold,
    otherwise returns *word* unchanged.
    """
    if not _HAS_RAPIDFUZZ or len(word) < 4:
        return word
        
    if word.lower() in _IGNORE_LIST:
        return word

    # Try general domain keywords first (high threshold)
    result_kw = _rfprocess.extractOne(
        word,
        _DOMAIN_KEYWORDS,
        scorer=_rffuzz.ratio,
        processor=lambda x: x.lower() if isinstance(x, str) else x,
        score_cutoff=85,
    )
    if result_kw:
        match, score, _ = result_kw
        log.debug(f"Fuzzy keyword: {word!r} -> {match!r} (score={score:.0f})")
        return match

    result = _rfprocess.extractOne(
        word,
        _DISTRICTS,
        scorer=_rffuzz.ratio,
        processor=lambda x: x.lower() if isinstance(x, str) else x,
        score_cutoff=_FUZZY_THRESHOLD,
    )
    if result:
        match, score, _ = result
        log.debug(f"Fuzzy district: {word!r} -> {match!r} (score={score:.0f})")
        return match
        
    if _VILLAGES and not word.lower().endswith('s'):
        result_vill = _rfprocess.extractOne(
            word,
            _VILLAGES,
            scorer=_rffuzz.ratio,
            processor=lambda x: x.lower() if isinstance(x, str) else x,
            score_cutoff=80,
        )
        if result_vill:
            match, score, _ = result_vill
            log.debug(f"Fuzzy village: {word!r} -> {match!r} (score={score:.0f})")
            return match

    return word


def normalize_query(query: str) -> str:
    """
    Apply two normalization passes to a raw user query:

    1. **Vocabulary correction** — exact case-insensitive replacements for
       known gender/caste/education/occupation synonyms.
    2. **Fuzzy district correction** — RapidFuzz WRatio against the canonical
       Rajasthan district list for words that look like district names.

    Args:
        query: Raw user natural language question.

    Returns:
        Normalized query string with corrections applied.
    """
    original = query

    # ── Pass 1: vocabulary map (whole-word, case-insensitive) ────────────────
    for wrong, right in _VOCAB_MAP.items():
        query = re.sub(
            rf"\b{re.escape(wrong)}\b",
            right,
            query,
            flags=re.IGNORECASE,
        )

    # ── Pass 2: per-word fuzzy district correction ───────────────────────────
    # Only try words that are title-case or ≥ 5 chars (likely proper nouns)
    words = query.split()
    corrected_words = []
    for word in words:
        # Strip punctuation for matching, reattach after
        stripped = re.sub(r"[^\w\-]", "", word)
        if len(stripped) >= 5:
            candidate = _fuzzy_correct_word(stripped)
            word = word.replace(stripped, candidate)
        corrected_words.append(word)
    query = " ".join(corrected_words)

    if query != original:
        log.info(f"Normalized: {original!r} -> {query!r}")

    return query
