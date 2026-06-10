"""
rag/domain_dict.py — Phase 2b: Regex-based domain dictionary.

Maps informal/colloquial user phrasing to exact SQL fragments using
whole-word regex matching. These hints are injected into the LLM prompt
so the model knows the exact column values to use.

Design principles:
- Word-boundary anchored (\b) to avoid false positives
- Case-insensitive matching
- Returns a list of SQL hint strings to append to the prompt
- Zero LLM involvement — pure deterministic regex

Usage:
    from rag.domain_dict import extract_sql_hints
    hints = extract_sql_hints("show poor families in rural areas")
    # -> ["IS_RURAL = '1' (rural area)", "INCOME < 50000 (low income / poor)"]
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger

log = get_logger("nl2sql.domain_dict")


@dataclass
class DomainRule:
    """A single domain mapping rule."""
    pattern: str          # regex pattern (will be compiled with IGNORECASE | UNICODE)
    sql_hint: str         # exact SQL fragment / description to inject
    description: str      # human-readable explanation for logging


# ── Master domain rules ───────────────────────────────────────────────────────
# Order matters: more specific rules first
_RULES: list[DomainRule] = [

    # ── Geography ─────────────────────────────────────────────────────────────
    DomainRule(
        r"\b(rural|village|gram|gaon|panchayat)\b",
        "IS_RURAL = '1'",
        "rural area indicator",
    ),
    DomainRule(
        r"\b(urban|city|town|ward|municipal|nagar)\b",
        "IS_RURAL = '0'",
        "urban area indicator",
    ),

    # ── Caste categories ──────────────────────────────────────────────────────
    DomainRule(
        r"\b(scheduled caste|sc caste|dalit)\b",
        "CASTE_CATEGORY = 'SC'",
        "Scheduled Caste",
    ),
    DomainRule(
        r"\b(scheduled tribe|st caste|tribal|adivasi)\b",
        "CASTE_CATEGORY = 'ST'",
        "Scheduled Tribe",
    ),
    DomainRule(
        r"\b(obc|other backward class(es)?|backward class(es)?)\b",
        "CASTE_CATEGORY = 'OBC'",
        "Other Backward Class",
    ),
    DomainRule(
        r"\b(general category|gen category|unreserved|open category)\b",
        "CASTE_CATEGORY = 'GEN'",
        "General category",
    ),

    # ── Gender ────────────────────────────────────────────────────────────────
    DomainRule(
        r"\b(female|woman|women|girl|girls|mahila)\b",
        "GENDER = 'Female'",
        "female members",
    ),
    DomainRule(
        r"\b(male|man|men|boy|boys|purush)\b",
        "GENDER = 'Male'",
        "male members",
    ),

    # ── Age groups ────────────────────────────────────────────────────────────
    DomainRule(
        r"\b(senior citizen)s?\b",
        "AGE >= 60",
        "senior citizens (age >= 60)",
    ),
    DomainRule(
        r"\b(minor)s?|\b(child|children)\b",
        "AGE < 18",
        "minors (age < 18)",
    ),
    DomainRule(
        r"\b(youth|young|youngster|yuva)\b",
        "AGE BETWEEN 18 AND 35",
        "youth (18-35)",
    ),
    DomainRule(
        r"\b(working age|adult)s?\b",
        "AGE BETWEEN 18 AND 60",
        "working age adults",
    ),

    # ── Income / poverty ──────────────────────────────────────────────────────
    DomainRule(
        r"\b(poor|poverty|low income|bpl|below poverty)\b",
        "INCOME < 50000",
        "low income / poor families",
    ),
    DomainRule(
        r"\b(rich|wealthy|high income|affluent)\b",
        "INCOME > 500000",
        "high income families",
    ),
    DomainRule(
        r"\b(middle class|middle income|moderate income)\b",
        "INCOME BETWEEN 50000 AND 500000",
        "middle income range",
    ),
    DomainRule(
        r"\b(no income|zero income|unemployed income|income nil|without income|without any income)\b",
        "INCOME = 0",
        "zero income members",
    ),

    # ── Marital status ────────────────────────────────────────────────────────
    DomainRule(
        r"\b(widow|widower|widowed)s?\b",
        "MARITAL_STATUS = 'Widow'",
        "widowed members",
    ),
    DomainRule(
        r"\b(married|husband|wife|spouse|wedded)s?\b",
        "MARITAL_STATUS = 'Married'",
        "married members",
    ),
    DomainRule(
        r"\b(single|bachelor|spinster|never married|unmarried)s?\b",
        "MARITAL_STATUS = 'Unmarried'",
        "unmarried members",
    ),
    DomainRule(
        r"\b(divorced|separated)\b",
        "MARITAL_STATUS = 'Divorced'",
        "divorced members",
    ),

    # ── Education ─────────────────────────────────────────────────────────────
    DomainRule(
        r"\b(literate|educated|can read)\b",
        "EDUCATION != 'illiterate'",
        "literate members (anyone not illiterate)",
    ),
    DomainRule(
        r"\b(illiterate|uneducated|no education|cannot read)\b",
        "EDUCATION = 'illiterate'",
        "illiterate members",
    ),
    DomainRule(
        r"(?<!post )(?<!post-)\b(graduate|degree holder|ba|bsc|bcom|b\.a|b\.sc|b\.com)s?\b",
        "EDUCATION ILIKE '%graduate%'",
        "graduate-level education",
    ),
    DomainRule(
        r"\b(post.?graduate|pg |masters?|m\.a|m\.sc|m\.com|mba)\b",
        "EDUCATION ILIKE '%post%graduate%'",
        "post-graduate education",
    ),
    DomainRule(
        r"\b(10th|tenth|matriculate|ssc|secondary)(?:\s+pass(ed)?)?\b",
        "EDUCATION ILIKE '%10%'",
        "10th / secondary level",
    ),
    DomainRule(
        r"\b(12th|twelfth|intermediate|hsc|higher secondary|senior secondary)(?:\s+pass(ed)?)?\b",
        "EDUCATION ILIKE '%12%'",
        "12th / higher secondary level",
    ),

    DomainRule(
        r"\b(8th|eighth|middle school)(?:\s+pass(ed)?)?\b",
        "EDUCATION ILIKE '%8%'",
        "8th / middle level",
    ),
    DomainRule(
        r"\b(5th|fifth|primary school)(?:\s+pass(ed)?)?\b",
        "EDUCATION ILIKE '%5%'",
        "5th / primary level",
    ),

    # ── Occupation ────────────────────────────────────────────────────────────
    DomainRule(
        r"\b(farmer|kisan|agriculture|agri worker|cultivator)s?\b",
        "OCCUPATION ILIKE '%farmer%' OR OCCUPATION ILIKE '%agri%'",
        "farmers / agricultural workers",
    ),
    DomainRule(
        r"\b(labourer|laborer|labour|labor|daily wage|mazdoor)\b",
        "OCCUPATION ILIKE '%labour%' OR OCCUPATION ILIKE '%labor%'",
        "labourers / daily wage workers",
    ),
    DomainRule(
        r"\b(home.?maker|house.?wife|house.?wives|griha)s?\b",
        "OCCUPATION ILIKE '%home%maker%' OR OCCUPATION ILIKE '%housewife%'",
        "homemakers",
    ),
    DomainRule(
        r"\b(unemployed|jobless|out of work|not working)\b",
        "OCCUPATION ILIKE '%unemployed%'",
        "unemployed members",
    ),
    DomainRule(
        r"\b(student|pupil|studying|learner|scholar)s?\b",
        "OCCUPATION ILIKE '%student%'",
        "students",
    ),
    DomainRule(
        r"\b(government|state personnel|govt|gov|public sector|civil servant)s?\b",
        "OCCUPATION ILIKE '%state personnel%' OR OCCUPATION ILIKE '%autonomous%' OR OCCUPATION ILIKE '%psu%'",
        "government / state employees",
    ),
    DomainRule(
        r"\b(self.?employed|freelance|business|businessman|entrepreneur)s?\b",
        "OCCUPATION ILIKE '%self%employed%' OR OCCUPATION ILIKE '%business%'",
        "self-employed / businessmen",
    ),
    DomainRule(
        r"\b(contract|contractual)s?\b",
        "OCCUPATION ILIKE '%contract%'",
        "contractual employees",
    ),
    DomainRule(
        r"\b(banker|bank employee|psu)\b",
        "OCCUPATION ILIKE '%psu%bank%'",
        "bank / PSU employees",
    ),

    # ── Minority ─────────────────────────────────────────────────────────
    DomainRule(
        r"\b(minorities|minority)\b",
        "MINORITY IS NOT NULL",
        "minority community members",
    ),

    # ── Household / family ────────────────────────────────────────────────────
    DomainRule(
        r"\b(head of (family|household)|hof|householder|mukhiya)\b",
        "MEM_TYPE = 'HOF'",
        "head of family",
    ),
    DomainRule(
        r"\b(large family|big family|large household)\b",
        "/* households with many members — use GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 5 */",
        "large household hint",
    ),

    # ── Banking — abbreviation expansion ────────────────────────────────────
    # The DB stores full official bank names (e.g. 'STATE BANK OF INDIA').
    # Users naturally use abbreviations (SBI, PNB, BOB, etc.).
    # Each rule maps the abbreviation to the exact ILIKE/IFSC pattern.
    DomainRule(
        r"\b(sbi|state bank of india|state bank)\b",
        "(BANK ILIKE '%State Bank of India%' OR IFSC_CODE ILIKE 'SBIN%')",
        "State Bank of India",
    ),
    DomainRule(
        r"\b(pnb|punjab national bank)\b",
        "(BANK ILIKE '%Punjab National Bank%' OR IFSC_CODE ILIKE 'PUNB%')",
        "Punjab National Bank",
    ),
    DomainRule(
        r"\b(bob|bank of baroda)\b",
        "(BANK ILIKE '%Bank of Baroda%' OR IFSC_CODE ILIKE 'BARB%')",
        "Bank of Baroda",
    ),
    DomainRule(
        r"\b(rgb|rajasthan gramin bank|gramin bank)\b",
        "BANK ILIKE '%Rajasthan Gramin Bank%'",
        "Rajasthan Gramin Bank",
    ),
    DomainRule(
        r"\b(uco bank|uco)\b",
        "(BANK ILIKE '%UCO Bank%' OR IFSC_CODE ILIKE 'UCBA%')",
        "UCO Bank",
    ),
    DomainRule(
        r"\b(central bank of india|central bank)\b",
        "BANK ILIKE '%Central Bank of India%'",
        "Central Bank of India",
    ),
    DomainRule(
        r"\b(union bank of india|union bank)\b",
        "(BANK ILIKE '%Union Bank%' OR IFSC_CODE ILIKE 'UBIN%')",
        "Union Bank of India",
    ),
    DomainRule(
        r"\b(icici)\b",
        "(BANK ILIKE '%ICICI%' OR IFSC_CODE ILIKE 'ICIC%')",
        "ICICI Bank",
    ),
    DomainRule(
        r"\b(boi|bank of india)\b",
        "(BANK ILIKE '%Bank of India%' OR IFSC_CODE ILIKE 'BKID%')",
        "Bank of India",
    ),
    DomainRule(
        r"\b(canara bank|canara)\b",
        "(BANK ILIKE '%Canara Bank%' OR IFSC_CODE ILIKE 'CNRB%')",
        "Canara Bank",
    ),
    DomainRule(
        r"\b(idbi)\b",
        "(BANK ILIKE '%IDBI%' OR IFSC_CODE ILIKE 'IBKL%')",
        "IDBI Bank",
    ),
    DomainRule(
        r"\b(axis bank|axis)\b",
        "(BANK ILIKE '%Axis Bank%' OR IFSC_CODE ILIKE 'UTIB%')",
        "Axis Bank",
    ),
    DomainRule(
        r"\b(hdfc)\b",
        "(BANK ILIKE '%HDFC%' OR IFSC_CODE ILIKE 'HDFC%')",
        "HDFC Bank",
    ),
    DomainRule(
        r"\b(indusind|indus ind)\b",
        "(BANK ILIKE '%IndusInd%' OR IFSC_CODE ILIKE 'INDB%')",
        "IndusInd Bank",
    ),
    DomainRule(
        r"\b(indian bank)\b",
        "(BANK ILIKE '%Indian Bank%' OR IFSC_CODE ILIKE 'IDIB%')",
        "Indian Bank",
    ),
    DomainRule(
        r"\b(bank of maharashtra|bom)\b",
        "BANK ILIKE '%Bank of Maharashtra%'",
        "Bank of Maharashtra",
    ),
    DomainRule(
        r"\b(punjab and sind bank|psb)\b",
        "BANK ILIKE '%Punjab and Sind%'",
        "Punjab and Sind Bank",
    ),
    DomainRule(
        r"\b(au small finance|au bank)\b",
        "BANK ILIKE '%AU Small Finance%'",
        "AU Small Finance Bank",
    ),
    DomainRule(
        r"\b(airtel payments bank|airtel bank)\b",
        "BANK ILIKE '%Airtel Payments%'",
        "Airtel Payments Bank",
    ),
    DomainRule(
        r"\b(fino payments bank|fino)\b",
        "BANK ILIKE '%Fino Payments%'",
        "Fino Payments Bank",
    ),
    DomainRule(
        r"\b(bandhan bank|bandhan)\b",
        "BANK ILIKE '%Bandhan Bank%'",
        "Bandhan Bank",
    ),
    DomainRule(
        r"\b(brkgb|baroda rajasthan|kshetriya gramin)\b",
        "BANK ILIKE '%Baroda Rajasthan%'",
        "Baroda Rajasthan Kshetriya Gramin Bank",
    ),
    DomainRule(
        r"\b(cooperative bank|co-operative bank|co operative bank)\b",
        "BANK ILIKE '%Co-Operative Bank%' OR BANK ILIKE '%Cooperative Bank%'",
        "cooperative bank",
    ),
    # No bank account
    DomainRule(
        r"\b(no bank|unbanked|without (?:a )?(?:bank )?account|without bank|no account|no banking|(?:don'?t|do not) have a bank account)\b",
        "(ACCOUNT_NO IS NULL OR ACCOUNT_NO = '' OR BANK IS NULL OR BANK = '')",
        "members without bank account",
    ),
    # Has bank account
    DomainRule(
        r"\b(with (?:a )?(?:bank )?account|has (?:a )?(?:bank )?account|banked)\b",
        "(ACCOUNT_NO IS NOT NULL AND ACCOUNT_NO != '' AND BANK IS NOT NULL AND BANK != '')",
        "members with bank account",
    ),
]

# Pre-compile all patterns for speed
_COMPILED: list[tuple[re.Pattern, DomainRule]] = [
    (re.compile(r.pattern, re.IGNORECASE | re.UNICODE), r)
    for r in _RULES
]

import json

# Global dictionary to hold dynamic caste SQL hints
# Key: canonical caste name, Value: SQL IN clause string
_CASTE_HINTS: dict[str, str] = {}
_HAS_RAPIDFUZZ = False
try:
    from rapidfuzz import process as _rfprocess, fuzz as _rffuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    pass

def _load_caste_mappings() -> None:
    mapping_file = Path(__file__).resolve().parents[1] / "data" / "caste_mapping.json"
    if not mapping_file.exists():
        return
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        for canonical, raw_list in mapping.items():
            if not raw_list:
                continue
            # Build SQL IN clause
            escaped_list = [c.replace("'", "''") for c in raw_list]
            in_clause = ", ".join(f"'{c}'" for c in escaped_list)
            
            # Also build ILIKE clauses for NAME_EN (skip hindi for ILIKE as it's NAME_EN)
            ilike_clauses = [f"NAME_EN ILIKE '%{c}%'" for c in escaped_list if c.isascii()]
            
            if ilike_clauses:
                ilike_str = " OR " + " OR ".join(ilike_clauses)
                sql_hint = f"(CASTE IN ({in_clause}){ilike_str})"
            else:
                sql_hint = f"CASTE IN ({in_clause})"
                
            import re
            # Add canonical name as a trigger
            canon = canonical.lower().strip()
            if len(canon) >= 3:
                _CASTE_HINTS[canon] = sql_hint
                if not canon.endswith('s'):
                    _CASTE_HINTS[canon + 's'] = sql_hint
            
            # Add all known variations (including Devanagari) as triggers mapping to the FULL list
            for c in raw_list:
                # Remove parenthesized text (e.g. "(MUSLIM)")
                clean_c = re.sub(r"\(.*?\)", "", c)
                # Remove leading numbers (e.g. "48 ")
                clean_c = re.sub(r"^\d+\s*", "", clean_c)
                
                # remove non-alphanumeric (including unicode letters) but KEEP spaces
                trigger = re.sub(r"[^\w\-\s]", "", clean_c).lower().strip()
                
                if len(trigger) >= 3:
                    if trigger not in _CASTE_HINTS:
                        _CASTE_HINTS[trigger] = sql_hint
                    if not trigger.endswith('s'):
                        if (trigger + 's') not in _CASTE_HINTS:
                            _CASTE_HINTS[trigger + 's'] = sql_hint
                            
                    # Also add individual words if there are multiple words (e.g. "dasnam" from "dasnam gauswami")
                    for w in trigger.split():
                        if len(w) >= 4:
                            if w not in _CASTE_HINTS:
                                _CASTE_HINTS[w] = sql_hint
                            if not w.endswith('s') and (w + 's') not in _CASTE_HINTS:
                                _CASTE_HINTS[w + 's'] = sql_hint
    except Exception as e:
        log.error(f"Failed to load caste mappings: {e}")

_load_caste_mappings()


def extract_sql_hints(query: str) -> tuple[list[str], set[str]]:
    """
    Scan *query* for domain-specific phrases and return a deduplicated list
    of SQL hint strings that the LLM should incorporate, plus the set of words mapped.
    """
    hints: list[str] = []
    seen: set[str] = set()
    dynamic_words: set[str] = set()
    mapped_words: set[str] = set()

    def add_hint(h: str, desc: str, matched_text: str = ""):
        if matched_text:
            # Strip punctuation and add lowercase words to mapped_words
            words = re.sub(r"[^\w\-\s]", "", matched_text).lower().split()
            mapped_words.update(words)
        if h not in seen:
            hints.append(h)
            seen.add(h)
            log.info(f"Dynamic hint: {desc!r} -> {h!r}")

    # ── 1. Dynamic Extractors (Age, Income, Relational, Unmapped Surnames) ────
    # Age Filters
    # The negative lookahead prevents matching income ("over 50k") or counts ("over 5 members")
    age_neg_lookahead = r"(?!\s*(?:k|l|m|cr|thousand|lakh|lac|crore|million|member|people|person|household))"
    for m in re.finditer(rf"\b(?:age\s+)?between\s+(\d{{1,3}})\s+and\s+(\d{{1,3}})\b{age_neg_lookahead}", query, re.I):
        add_hint(f"AGE BETWEEN {m.group(1)} AND {m.group(2)}", "age between", m.group(0))
    for m in re.finditer(rf"\b(?:(?:age\s+)?(greater than|more than|>|above|over|>=)|(greater than|more than|>|above|over|>=)\s+age)\s+(\d{{1,3}})\b{age_neg_lookahead}", query, re.I):
        add_hint(f"AGE > {m.group(3)}", "age >", m.group(0))
    for m in re.finditer(rf"\b(?:(?:age\s+)?(less than|under|<|below|<=)|(less than|under|<|below|<=)\s+age)\s+(\d{{1,3}})\b{age_neg_lookahead}", query, re.I):
        add_hint(f"AGE < {m.group(3)}", "age <", m.group(0))
        
    # Income Filters
    def parse_money(val_str: str, unit_str: str | None) -> int:
        val = int(val_str)
        if not unit_str:
            return val
        u = unit_str.lower()
        if u in ("k", "thousand", "thousands"): return val * 1000
        if u in ("l", "lakh", "lakhs", "lac", "lacs"): return val * 100000
        if u in ("cr", "crore", "crores"): return val * 10000000
        if u in ("m", "million", "millions"): return val * 1000000
        return val

    unit_pat = r"(?:\s*(k|l|m|cr|thousands?|lakhs?|lacs?|crores?|millions?))"
    
    for m in re.finditer(rf"\b(?:income|salary|earning|earn)s?\s+between\s+(\d+){unit_pat}?\s+and\s+(\d+){unit_pat}?\b", query, re.I):
        v1 = parse_money(m.group(1), m.group(2))
        v2 = parse_money(m.group(3), m.group(4))
        add_hint(f"INCOME BETWEEN {v1} AND {v2}", "income between", m.group(0))
        
    for m in re.finditer(rf"\b(?:income|salary|earning|earn)s?\s+(greater than|more than|>|above|over|>=)\s+(\d+){unit_pat}?\b", query, re.I):
        op = ">=" if ">=" in m.group(1) else ">"
        val = parse_money(m.group(2), m.group(3))
        add_hint(f"INCOME {op} {val}", "income >", m.group(0))
        
    for m in re.finditer(rf"\b(?:income|salary|earning|earn)s?\s+(less than|under|<|below|<=)\s+(\d+){unit_pat}?\b", query, re.I):
        op = "<=" if "<=" in m.group(1) else "<"
        val = parse_money(m.group(2), m.group(3))
        add_hint(f"INCOME {op} {val}", "income <", m.group(0))
        
    # Multi-Location OR has been delegated to LLM to prevent false positives with explicit geographical mapping.
    
    # Unmapped Surnames are now handled by LLM directly to prevent false positives.

    # ── 2. Static Rules ───────────────────────────────────────────────────────
    for pattern, rule in _COMPILED:
        m = pattern.search(query)
        if m:
            if rule.sql_hint not in seen:
                hints.append(rule.sql_hint)
                seen.add(rule.sql_hint)
                words = re.sub(r"[^\w\-\s]", "", m.group(0)).lower().split()
                mapped_words.update(words)
                log.debug(f"Domain hit: {rule.description!r} -> {rule.sql_hint!r}")

    if _HAS_RAPIDFUZZ and _CASTE_HINTS:
        stopwords = {"give", "me", "data", "of", "all", "people", "from", "caste", "if", "show", "count", "list", "how", "many", "the", "in", "for", "with", "who", "are", "is", "what", "where", "and", "or", "than", "more", "less", "under", "over", "don't", "dont", "person"}
        words = re.sub(r"[^\w\-\s]", "", query).lower().split()
        ngrams = []
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram_words = words[i:i+n]
                ngram = " ".join(ngram_words)
                # Filter out numbers, pure stopwords, and ANY word that was already matched as a dynamic name
                if len(ngram) > 2 and not all(w in stopwords for w in ngram_words) and not any(c.isdigit() for c in ngram):
                    if not any(w in dynamic_words for w in ngram_words):
                        ngrams.append(ngram)
        
        for ngram in ngrams:
            result = _rfprocess.extractOne(
                ngram,
                _CASTE_HINTS.keys(),
                scorer=_rffuzz.ratio,
                score_cutoff=85
            )
            if result:
                match_key, score, _ = result
                hint = _CASTE_HINTS[match_key]
                words = re.sub(r"[^\w\-\s]", "", ngram).lower().split()
                mapped_words.update(words)
                if hint not in seen:
                    log.info(f"Fuzzy caste hit: {ngram!r} -> {match_key!r} ({score}) -> {hint!r}")
                    hints.append(hint)
                    seen.add(hint)

    # ── 3. Exact Geographic Mapping (Districts and Villages) ─────────────────
    from rag.normalizer import _DISTRICTS, _VILLAGES
    q_words = [re.sub(r"[^\w\-\s]", "", w).lower() for w in query.split()]
    for d in _DISTRICTS:
        if d.lower() in q_words:
            add_hint(f"DISTRICT_NAME_ENG = '{d}'", "exact district", d)
            mapped_words.update(d.lower().split())
    for v in _VILLAGES:
        if v.lower() in q_words:
            add_hint(f"VILL_NAME_ENG = '{v}'", "exact village", v)
            mapped_words.update(v.lower().split())

    # ── 4. Explicit Name Mapping ─────────────────────────────────────────────
    for m in re.finditer(r"\b(?:named|name is|surname is|called)\s+([a-zA-Z]+)\b", query, re.I):
        name = m.group(1).replace("'", "''")
        add_hint(f"(NAME_EN ILIKE '%{name}%' OR CASTE ILIKE '%{name}%')", "explicit name", m.group(0))

    # ── 5. Ambiguous Single Location ──────────────────────────────────────────
    # Ambiguous Single Location has been fully delegated to the LLM Prompt Rule to avoid
    # interfering with explicit Geographic Static Matches (e.g. "Bikaner").

    if hints:
        log.info(f"Domain hints ({len(hints)}): {hints}")

    return hints, mapped_words


def format_hints_for_prompt(hints: list[str]) -> str:
    """
    Format extracted hints as a block to inject into the LLM prompt.

    Returns empty string if no hints.
    """
    if not hints:
        return ""

    lines = ["-- Domain hints (use these exact conditions where applicable):"]
    for h in hints:
        lines.append(f"--   {h}")
    return "\n".join(lines)
