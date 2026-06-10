"""
llm/fast_sql.py — Deterministic SQL builder + Scaffold generator.

Three modes of operation:
  1. Full Deterministic  — entire SQL built without LLM (<2ms).
  2. Scaffold-Assisted   — WHERE clause pre-built, LLM fills remaining gaps.
  3. Full LLM            — unknown query, no scaffold possible.

Covers:
 - Simple SELECT with known WHERE conditions
 - COUNT / AVG / SUM aggregations
 - GROUP BY aggregations (by district, caste, gender, etc.)
 - Family-size / household queries
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TABLE_NAME

# Columns to show by default for member listing
_DEFAULT_COLS = (
    "NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, "
    "VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS"
)


@dataclass
class PartialScaffold:
    """
    Represents a partially-built SQL query where the deterministic layer
    has pre-computed the WHERE conditions it understands, and signals to
    the LLM what it still needs to figure out.
    """
    # All known WHERE conditions (exact SQL fragments)
    known_conditions: list[str] = field(default_factory=list)
    # Tokens from the query that were NOT mapped — the LLM must handle these
    unmapped_tokens: list[str] = field(default_factory=list)
    # Query intent flags
    is_count: bool = False
    is_list: bool = False
    groupby_col: str | None = None
    # Pre-built WHERE clause string (from known_conditions)
    where_clause: str = ""
    # Which columns are already covered (so schema can be pruned)
    covered_columns: set[str] = field(default_factory=set)

# ─────────────────────────────────────────────────────────────────────────────
# Pattern detectors (order matters — most specific first)
# ─────────────────────────────────────────────────────────────────────────────

def _is_count(q: str) -> bool:
    return bool(re.search(
        r"\b(how many|count|total number|number of|tally)\b", q, re.I
    ))

def _is_list(q: str) -> bool:
    return bool(re.search(
        r"\b(show|list|display|find|get|who|which member|members?|children|son|daughter|wife|husband|spouse|families|family|details|give|tell|people|person|males?|females?|men|women|adults?|kids?|boys?|girls?)\b", q, re.I
    ))

def _groupby_col(q: str) -> str | None:
    """Return a GROUP BY column if the question asks for breakdown."""
    patterns = [
        (r"\bby district\b", "DISTRICT_NAME_ENG"),
        (r"\bper district\b", "DISTRICT_NAME_ENG"),
        (r"\bdistrict.?wise\b", "DISTRICT_NAME_ENG"),
        (r"\bby caste\b", "CASTE_CATEGORY"),
        (r"\bper caste\b", "CASTE_CATEGORY"),
        (r"\bcaste.?wise\b", "CASTE_CATEGORY"),
        (r"\bby gender\b", "GENDER"),
        (r"\bper gender\b", "GENDER"),
        (r"\bgender.?wise\b", "GENDER"),
        (r"\bby occupation\b", "OCCUPATION"),
        (r"\bper occupation\b", "OCCUPATION"),
        (r"\bby education\b", "EDUCATION"),
        (r"\bper education\b", "EDUCATION"),
        (r"\bby marital\b", "MARITAL_STATUS"),
        (r"\bby bank\b", "BANK"),
        (r"\bby village\b", "VILL_NAME_ENG"),
        (r"\bby block\b", "BLOCK_NAME_ENG"),
        (r"\bby rural.?urban\b", "IS_RURAL"),
    ]
    for pat, col in patterns:
        if re.search(pat, q, re.I):
            return col
    return None

def _avg_col(q: str) -> str | None:
    for pat, col in [
        (r"\baverage\s+(?:income|salary|earning)\b", "INCOME"),
        (r"\bavg\s+(?:income|salary)\b", "INCOME"),
        (r"\baverage\s+age\b", "AGE"),
        (r"\bavg\s+age\b", "AGE"),
    ]:
        if re.search(pat, q, re.I):
            return col
    return None

def _sum_col(q: str) -> str | None:
    for pat, col in [
        (r"\btotal\s+income\b", "INCOME"),
        (r"\bsum\s+(?:of\s+)?income\b", "INCOME"),
    ]:
        if re.search(pat, q, re.I):
            return col
    return None

def _family_size_op(q: str) -> tuple[str, int] | None:
    # Handle specific case: "N or more"
    m_1 = re.search(r"(?:households?|famil(?:y|ies))\s+with\s+(\d+)\s+or\s+(?:more|greater)", q, re.I)
    if m_1:
        return ">=", int(m_1.group(1))

    m = re.search(
        r"(?:households?|famil(?:y|ies))\s+with\s+"
        r"(more than|greater than|over|above|at least|minimum|at most|no more than|exactly|fewer than|less than|under|below)\s+(\d+)\s+(?:members?|people|persons?|children|kids|adults|males|females)",
        q, re.I
    )
    if not m:
        return None
    word, n = m.group(1).lower(), int(m.group(2))
    op_map = {
        "more than": ">", "greater than": ">", "over": ">", "above": ">",
        "at least": ">=", "minimum": ">=",
        "at most": "<=", "no more than": "<=",
        "exactly": "=",
        "fewer than": "<", "less than": "<", "under": "<", "below": "<",
    }
    op = op_map.get(word, ">")
    return op, n

def _household_count_op(q: str) -> tuple[str, int] | None:
    """Detect 'how many households/families have more/less/exactly N members'."""
    m_1 = re.search(r"how many\s+(?:households?|famil(?:y|ies))\s+(?:have|with|has)\s+(\d+)\s+or\s+(?:more|greater)", q, re.I)
    if m_1:
        return ">=", int(m_1.group(1))

    m = re.search(
        r"how many\s+(?:households?|famil(?:y|ies))\s+(?:have|with|has)\s+"
        r"(more than|greater than|over|above|at least|at most|no more than|exactly|fewer than|less than|under|below)\s+(\d+)\s+(?:members?|people|persons?|children|kids|adults|males|females)",
        q, re.I
    )
    if not m:
        return None
    word, n = m.group(1).lower(), int(m.group(2))
    op_map = {
        "more than": ">", "greater than": ">", "over": ">", "above": ">",
        "at least": ">=", "at most": "<=", "no more than": "<=",
        "exactly": "=", "fewer than": "<", "less than": "<",
        "under": "<", "below": "<",
    }
    op = op_map.get(word, ">")
    return op, n

def _largest_family(q: str) -> bool:
    return bool(re.search(
        r"(biggest|largest|most members?|maximum members?|which family has the most)",
        q, re.I
    ))

def _smallest_family(q: str) -> bool:
    return bool(re.search(
        r"(smallest|fewest members?|minimum members?|which family has the least)",
        q, re.I
    ))

def _extreme_col(q: str) -> tuple[str, str] | None:
    """Detects MIN/MAX questions and returns (COLUMN_NAME, ASC|DESC)."""
    if re.search(r"\b(oldest|eldest|maximum age)\b", q, re.I):
        return ("AGE", "DESC")
    if re.search(r"\b(youngest|minimum age)\b", q, re.I):
        return ("AGE", "ASC")
    if re.search(r"\b(highest income|richest|most income|maximum income)\b", q, re.I):
        return ("INCOME", "DESC")
    if re.search(r"\b(lowest income|poorest|least income|minimum income)\b", q, re.I):
        return ("INCOME", "ASC")
    return None

def _get_unmapped_words(query: str, mapped_words: set[str]) -> list[str]:
    """
    Checks if the user's query contains any significant words that weren't successfully
    mapped by the domain dictionary. Returns the list of unmapped words.
    """
    stopwords = {
        # UI verbs / action words
        "show", "list", "display", "find", "get", "who", "which", "give", "me", "tell",
        "fetch", "return", "retrieve", "provide", "output", "count",
        # Articles / connectives
        "all", "the", "in", "for", "with", "are", "is", "what", "where",
        "and", "or", "than", "more", "less", "under", "over", "don't", "dont",
        "of", "from", "have", "has", "a", "an", "to", "by", "that", "those", "these",
        # Common domain nouns that don't add filtering value
        "data", "people", "person", "members", "member", "details", "information", 
        "records", "number", "family", "families", "caste", "account", "bank", "district", "village", "category",
        "households", "household", "children", "kids", "adults",
        "head", "heads",
        # Aggregate keywords
        "total", "how", "many", "average", "avg", "sum",
        # Geography filler
        "area", "areas", "city", "town", "region", "regions",
        "zone", "zones", "place", "places", "location", "locations",
        # Generic domain verbs / nouns
        "earn", "earning", "earns", "belong", "belongs", "belonging", "live",
        "living", "lives", "reside", "residing", "resides", "work", "working",
        "name", "named", "called", "whose", "having", "surname", "first", "last",
        # Quantifier fillers (handled by regex)
        "above", "below", "between", "greater", "less", "at", "least",
        "most", "minimum", "maximum", "exactly", "up", "no", "only", "just",
        "did", "be", "been", "was", "were"
    }
    
    words = re.sub(r"[^\w\-\s]", "", query).lower().split()
    
    unmapped = []
    for w in words:
        if w in stopwords:
            continue
        if w.isdigit():
            continue
        if w in mapped_words:
            continue
            
        # Leftover unmapped noun/adjective/operator!
        unmapped.append(w)
        
    return unmapped

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _group_domain_hints(hints: list[str]) -> list[str]:
    from collections import defaultdict
    groups = defaultdict(list)
    for h in hints:
        m = re.match(r"^\s*\(?([A-Z_]+)\b", h)
        if m:
            col = m.group(1)
            groups[col].append(h)
        else:
            groups["OTHER"].append(h)
            
    final_hints = []
    for col, vals in groups.items():
        if col == "OTHER":
            final_hints.extend(vals)
        elif len(vals) == 1:
            final_hints.append(vals[0])
        else:
            if col in ["AGE", "INCOME"]:
                joined = " AND ".join(f"({v})" if " OR " in v and not v.startswith("(") else v for v in vals)
                final_hints.append(joined)
            else:
                joined = " OR ".join(f"({v})" if " AND " in v and not v.startswith("(") else v for v in vals)
                final_hints.append(f"({joined})")
                
    return final_hints

def try_build_sql(question: str, domain_hints: list[str], mapped_words: set[str] = None) -> str | None:
    """
    Attempt to build deterministic SQL from the question + domain hints.
    Returns SQL string on success, None if we should fall back to the LLM.

    Rules:
    - We ONLY bypass the LLM if domain_hints is non-empty OR the query is
      a pure structural query (total count, group-by with no filters).
    - If there's any ambiguity, return None → let the LLM handle it.
    """
    q = question.strip()
    
    has_unmapped = False
    unmapped_words = []
    if mapped_words is not None:
        unmapped_words = _get_unmapped_words(q, mapped_words)
        has_unmapped = len(unmapped_words) > 0

    # ── WHERE clause from domain hints ────────────────────────────────────────
    where = ""
    if domain_hints:
        grouped_hints = _group_domain_hints(domain_hints)
        where = " AND ".join(f"({h})" if " OR " in h and not h.startswith("(") else h
                             for h in grouped_hints)

    # ── Relational Query Deterministic Fallback ───────────────────────────────
    relational_targets = {
        "sons": "Son", "son": "Son",
        "daughters": "Daughter", "daughter": "Daughter",
        "wives": "Wife", "wife": "Wife",
        "husband": "Husband", "spouse": "Wife' OR RELATION_WITH_HOF ILIKE '%Husband",
        "children": "Son' OR RELATION_WITH_HOF ILIKE '%Daughter",
        "families": "ALL", "family": "ALL", "households": "ALL", "household": "ALL"
    }
    q_words = set(re.sub(r"[^\w\-\s]", "", q).lower().split())
    target_rel = None
    for word, rel in relational_targets.items():
        if word in q_words:
            target_rel = rel
            # Consider this relational keyword "mapped" so we don't fall back if it was the only unmapped word
            if word in unmapped_words:
                unmapped_words.remove(word)
            has_unmapped = len(unmapped_words) > 0
            break

    if target_rel:
        if target_rel == "ALL":
            where_clause = f"WHERE ENROLLMENT_ID IN (\n    SELECT ENROLLMENT_ID FROM {TABLE_NAME} WHERE {where}\n)" if where else ""
        else:
            where_clause = f"WHERE (RELATION_WITH_HOF ILIKE '%{target_rel}%')"
            if where:
                where_clause += f" AND ENROLLMENT_ID IN (\n    SELECT ENROLLMENT_ID FROM {TABLE_NAME} WHERE {where}\n)"
    else:
        where_clause = f"WHERE {where}" if where else ""

    # ── 2. Largest / Smallest family ──────────────────────────────────────────
    if _largest_family(q):
        return (
            f"SELECT ENROLLMENT_ID, COUNT(*) AS member_count\n"
            f"FROM {TABLE_NAME}\n"
            f"GROUP BY ENROLLMENT_ID\n"
            f"ORDER BY member_count DESC\n"
            f"LIMIT 1"
        )
    if _smallest_family(q):
        return (
            f"SELECT ENROLLMENT_ID, COUNT(*) AS member_count\n"
            f"FROM {TABLE_NAME}\n"
            f"GROUP BY ENROLLMENT_ID\n"
            f"ORDER BY member_count ASC\n"
            f"LIMIT 1"
        )

    # ── 2.5 Extremes (Oldest, Youngest, Richest) ──────────────────────────────
    ext = _extreme_col(q)
    if ext:
        col, order = ext
        return (
            f"SELECT {_DEFAULT_COLS}\n"
            f"FROM {TABLE_NAME}\n"
            f"{where_clause}\n"
            f"ORDER BY {col} {order}\n"
            f"LIMIT 1"
        )

    # ── 8. Count by Household size: "how many households have > 5 members" ────
    hc = _household_count_op(q)
    if hc:
        op, n = hc
        return (
            f"SELECT COUNT(*) AS total_count FROM (\n"
            f"  SELECT ENROLLMENT_ID\n"
            f"  FROM {TABLE_NAME}\n"
            f"  {where_clause}\n"
            f"  GROUP BY ENROLLMENT_ID\n"
            f"  HAVING COUNT(*) {op} {n}\n"
            f") sub"
        )

    # ── 3. Family size list: "show families with > N members" ─────────────────
    fs = _family_size_op(q)
    if fs:
        op, n = fs
        return (
            f"SELECT ENROLLMENT_ID, COUNT(*) AS member_count\n"
            f"FROM {TABLE_NAME}\n"
            f"{where_clause}\n"
            f"GROUP BY ENROLLMENT_ID\n"
            f"HAVING COUNT(*) {op} {n}\n"
            f"ORDER BY member_count DESC\n"
            f"LIMIT 500"
        )

    # ── 4. Average ────────────────────────────────────────────────────────────
    avg_col = _avg_col(q)
    if avg_col:
        grp = _groupby_col(q)
        if grp:
            return (
                f"SELECT {grp}, ROUND(AVG({avg_col}), 2) AS avg_{avg_col.lower()}\n"
                f"FROM {TABLE_NAME}\n"
                f"{where_clause}\n"
                f"GROUP BY {grp}\n"
                f"ORDER BY avg_{avg_col.lower()} DESC"
            )
        return (
            f"SELECT ROUND(AVG({avg_col}), 2) AS avg_{avg_col.lower()}\n"
            f"FROM {TABLE_NAME}\n"
            f"{where_clause}"
        )

    # ── 5. Sum ────────────────────────────────────────────────────────────────
    sum_col = _sum_col(q)
    if sum_col:
        grp = _groupby_col(q)
        if grp:
            return (
                f"SELECT {grp}, SUM({sum_col}) AS total_{sum_col.lower()}\n"
                f"FROM {TABLE_NAME}\n"
                f"{where_clause}\n"
                f"GROUP BY {grp}\n"
                f"ORDER BY total_{sum_col.lower()} DESC"
            )
        return (
            f"SELECT SUM({sum_col}) AS total_{sum_col.lower()}\n"
            f"FROM {TABLE_NAME}\n"
            f"{where_clause}"
        )

    # ── 6. Count queries ──────────────────────────────────────────────────────
    if _is_count(q):
        grp = _groupby_col(q)
        if grp:
            # Count by group — no LLM needed
            return (
                f"SELECT {grp}, COUNT(*) AS member_count\n"
                f"FROM {TABLE_NAME}\n"
                f"{where_clause}\n"
                f"GROUP BY {grp}\n"
                f"ORDER BY member_count DESC"
            )
        # Simple count — only bypass if we have a WHERE clause OR it's truly "total"
        if where or re.search(r"\b(total|all)\b", q, re.I):
            return (
                f"SELECT COUNT(*) AS total_members\n"
                f"FROM {TABLE_NAME}\n"
                f"{where_clause}"
            )

    # ── 7. List/show with clear domain hints ──────────────────────────────────
    if _is_list(q) and where:
        grp = _groupby_col(q)
        if not grp and not has_unmapped:
            # We are 100% confident every meaningful word was mapped to a hint.
            return (
                f"SELECT {_DEFAULT_COLS}\n"
                f"FROM {TABLE_NAME}\n"
                f"{where_clause}\n"
                f"LIMIT 500"
            )

    # ── 7.5 Generic Name/Caste search (FAST PATH) ─────────────────────────────
    if _is_list(q) and not _groupby_col(q) and has_unmapped:
        # Check if the unmapped words might just be a generic noun search
        # If the query contains complex relational intent, bail out to the LLM
        complex_words = {
            "not", "without", "except", "but", "only", "highest", "lowest", "top", "bottom",
            "who", "whose", "where", "families", "family", "households", "household", 
            "sons", "son", "daughters", "daughter", "wife", "husband", "spouse", "children"
        }
        q_words = set(re.sub(r"[^\w\-\s]", "", q).lower().split())
        
        if not (complex_words & q_words):
            unmapped_conditions = []
            q_lower = re.sub(r"[^\w\-\s]", "", q.lower()) # stripped version for regex
            
            for word in unmapped_words:
                term_escaped = word.replace("'", "''")
                is_loc = bool(re.search(rf"\b(from|in|village|city|district|place|lives in)\s+{re.escape(word)}\b", q_lower))
                is_name = bool(re.search(rf"\b(named|called|person|name|who is)\s+{re.escape(word)}\b", q_lower))
                is_caste = bool(re.search(rf"\b(caste|surname|community)\s+{re.escape(word)}\b", q_lower))
                
                if is_loc:
                    name_hint = (
                        f"(VILL_NAME_ENG ILIKE '%{term_escaped}%' OR GP_NAME_ENG ILIKE '%{term_escaped}%' "
                        f"OR BLOCK_NAME_ENG ILIKE '%{term_escaped}%' OR DISTRICT_NAME_ENG ILIKE '%{term_escaped}%')"
                    )
                elif is_name or is_caste:
                    name_hint = f"(NAME_EN ILIKE '%{term_escaped}%' OR CASTE ILIKE '%{term_escaped}%')"
                else:
                    name_hint = (
                        f"(CASTE ILIKE '%{term_escaped}%' OR NAME_EN ILIKE '%{term_escaped}%' "
                        f"OR VILL_NAME_ENG ILIKE '%{term_escaped}%' OR GP_NAME_ENG ILIKE '%{term_escaped}%' "
                        f"OR BLOCK_NAME_ENG ILIKE '%{term_escaped}%' OR DISTRICT_NAME_ENG ILIKE '%{term_escaped}%')"
                    )
                unmapped_conditions.append(name_hint)
            
            combined_unmapped = " AND ".join(f"({c})" for c in unmapped_conditions)
            final_where = f"{where_clause} AND ({combined_unmapped})" if where_clause else f"WHERE {combined_unmapped}"
            
            return (
                f"SELECT {_DEFAULT_COLS}\n"
                f"FROM {TABLE_NAME}\n"
                f"{final_where}\n"
                f"LIMIT 500;"
            )

    # ── 8. Pure group-by (no domain hints needed) ─────────────────────────────
    grp = _groupby_col(q)
    if grp and not where:
        return (
            f"SELECT {grp}, COUNT(*) AS member_count\n"
            f"FROM {TABLE_NAME}\n"
            f"GROUP BY {grp}\n"
            f"ORDER BY member_count DESC"
        )

    # Fall through → LLM needed
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold builder (Mode 2)
# ─────────────────────────────────────────────────────────────────────────────

# Map SQL hint prefixes to the columns they cover (for schema pruning)
_HINT_COLUMN_MAP: list[tuple[str, set[str]]] = [
    ("GENDER",          {"GENDER"}),
    ("IS_RURAL",        {"IS_RURAL"}),
    ("INCOME",          {"INCOME"}),
    ("AGE",             {"AGE"}),
    ("CASTE_CATEGORY",  {"CASTE_CATEGORY"}),
    ("MARITAL_STATUS",  {"MARITAL_STATUS"}),
    ("EDUCATION",       {"EDUCATION"}),
    ("OCCUPATION",      {"OCCUPATION"}),
    ("MEM_TYPE",        {"MEM_TYPE"}),
    ("MINORITY",        {"MINORITY"}),
    ("BANK",            {"BANK", "IFSC_CODE"}),
    ("ACCOUNT_NO",      {"ACCOUNT_NO", "BANK"}),
    ("DISTRICT_NAME",   {"DISTRICT_NAME_ENG"}),
    ("VILL_NAME",       {"VILL_NAME_ENG"}),
    ("CASTE",           {"CASTE"}),
    ("NAME_EN",         {"NAME_EN"}),
    ("FATHER_NAME",     {"FATHER_NAME_EN"}),
    ("MOTHER_NAME",     {"MOTHER_NAME_EN"}),
    ("SPOUSE_NAME",     {"SPOUSE_NAME_EN"}),
    ("ENROLLMENT_ID",   {"ENROLLMENT_ID"}),
]


def _columns_covered_by_hints(hints: list[str]) -> set[str]:
    """Return the set of column names already addressed by the given hints."""
    covered: set[str] = set()
    for hint in hints:
        h_upper = hint.upper()
        for prefix, cols in _HINT_COLUMN_MAP:
            if prefix in h_upper:
                covered.update(cols)
    return covered


def build_partial_scaffold(
    question: str,
    domain_hints: list[str],
    mapped_words: set[str] = None,
) -> PartialScaffold | None:
    """
    Builds a scaffolding structure to assist the LLM generation.
    - Resolves known domain hints into a WHERE clause prefix
    - Surfaces the unmapped tokens so the LLM knows what it still needs to handle
    - Records query intent (count, list, groupby) so the prompt builder can be smarter
    - Records which columns are already covered for schema pruning
    """
    if not domain_hints:
        return None  # nothing to pre-build — go full LLM

    q = question.strip()
    
    # If the query contains complex relational intent, bail out to the FULL LLM
    # so that it can use ENROLLMENT_ID subqueries. Scaffold forces a flat WHERE clause.
    complex_words = {
        "not", "without", "except", "but", "only", "highest", "lowest", "top", "bottom",
        "who", "whose", "where", "families", "family", "households", "household", 
        "sons", "son", "daughters", "daughter", "wife", "wives", "husband", "spouse", "children"
    }
    q_words = set(re.sub(r"[^\w\-\s]", "", q).lower().split())
    if complex_words & q_words:
        return None
        
    # Build the WHERE clause from all known hints
    grouped_hints = _group_domain_hints(domain_hints)
    where_parts = [
        f"({h})" if " OR " in h and not h.startswith("(") else h
        for h in grouped_hints
    ]
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Compute unmapped tokens (same stopwords as _has_unmapped_words)
    stopwords = {
        # UI verbs / action words
        "show", "list", "display", "find", "get", "who", "which", "give", "me", "tell",
        "fetch", "return", "retrieve", "provide", "output", "count",
        # Articles / connectives
        "all", "the", "in", "for", "with", "are", "is", "what", "where",
        "and", "or", "than", "more", "less", "under", "over", "don't", "dont",
        "of", "from", "have", "has", "a", "an", "to", "by", "that", "those", "these",
        # Common domain nouns that don't add filtering value
        "data", "people", "person", "members", "member", "details", "information", 
        "records", "number", "family", "families", "caste", "account", "bank", "district", "village", "category",
        "households", "household", "children", "kids", "adults",
        "head", "heads",
        # Aggregate keywords
        "total", "how", "many", "average", "avg", "sum",
        # Geography filler
        "area", "areas", "city", "town", "region", "regions",
        "zone", "zones", "place", "places", "location", "locations",
        # Generic domain verbs
        "earn", "earning", "earns", "belong", "belongs", "belonging", "live",
        "living", "lives", "reside", "residing", "resides", "work", "working",
        # Quantifier fillers (handled by regex)
        "above", "below", "between", "greater", "less", "at", "least",
        "most", "minimum", "maximum", "exactly", "up", "no", "only", "just",
        "did", "be", "been", "was", "were"
    }
    words = re.sub(r"[^\w\-\s]", "", q).lower().split()
    unmapped = [
        w for w in words
        if w not in stopwords
        and not w.isdigit()
        and (mapped_words is None or w not in mapped_words)
    ]

    covered = _columns_covered_by_hints(domain_hints)

    return PartialScaffold(
        known_conditions=domain_hints,
        unmapped_tokens=unmapped,
        is_count=_is_count(q),
        is_list=_is_list(q),
        groupby_col=_groupby_col(q),
        where_clause=where_clause,
        covered_columns=covered,
    )
