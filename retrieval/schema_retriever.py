from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41, RELATIONSHIPS, TABLES
from embeddings.faiss_store import FaissSchemaStore


# ── Domain vocabulary ─────────────────────────────────────────────────────────
CASTE_CATEGORY_TERMS = {"category", "sc", "st", "obc", "general", "gen",
                         "scheduled", "backward", "forward", "unreserved"}
CASTE_DETAIL_TERMS   = {"caste", "community", "jati"}
CASTE_TERMS          = CASTE_DETAIL_TERMS | CASTE_CATEGORY_TERMS

BANK_TERMS = {
    "bank", "account", "ifsc", "dbt", "payment",
    "sbi", "pnb", "bob", "hdfc", "icici", "uco", "canara", "baroda",
    "gramin", "cooperative", "unbanked", "no bank", "without bank", "no account",
}
IDENTITY_TERMS   = {"aadhaar", "jan", "enrollment", "mobile", "phone"}
MINORITY_TERMS   = {"minority", "muslim", "muslims", "jain", "jains"}
EDUCATION_TERMS  = {"education", "qualification", "illiterate", "literate",
                    "graduate", "school", "pass", "matric", "intermediate"}
RURAL_TERMS      = {"rural", "urban", "village people", "city families"}
GEOGRAPHY_TERMS  = {d.lower() for d in RAJASTHAN_DISTRICTS_41} | {
    "district", "zilla", "block", "village", "ward", "panchayat",
}
LOCATION_PREPOSITIONS = {"in", "from", "at"}
LOCATION_STOPWORDS    = {
    "age", "years", "male", "female", "boys", "girls",
    "beneficiaries", "beneficiary", "pension", "scheme",
}

KNOWN_CASTES = {
    "jat", "arai", "fakir", "mina", "rajput", "rajpoot", "brahman", "brahmin",
    "gurjar", "gujar", "valmiki", "balmiki", "bairwa", "berwa", "bazigar",
    "dangi", "dhobi", "darzi", "daroga", "sindhi", "jain", "chhipa",
    "जाट", "राजपूत", "मीना", "महाजन", "बाजीगर", "ब्राह्मण",
}


@dataclass
class RetrievalResult:
    question: str
    tables: list[str]
    columns: list[str]
    relationships: list[dict[str, str]]
    documents: list[dict[str, Any]]
    confidence: float


class SchemaRetriever:
    def __init__(self, store: FaissSchemaStore | None = None):
        self.store = store or FaissSchemaStore()

    def retrieve(self, question: str, top_k: int = settings.retrieval_top_k) -> RetrievalResult:
        docs = self.store.search(question, top_k=top_k)

        question_lower  = question.lower()
        question_terms  = set(re.findall(r"[a-zA-Z0-9]+", question_lower))
        columns: set[str] = set()

        # ── Lexical match every column against the question ───────────────────
        for col in COLUMNS:
            lexical_terms = [col.column.replace("_", " "), *col.aliases, *col.sample_values]
            if any(t and _matches(t, question_lower, question_terms) for t in lexical_terms):
                columns.add(col.qualified_name)

        # ── Semantic top-k (up to 6 extras) ──────────────────────────────────
        sem_added = 0
        for doc in docs:
            if doc.get("kind") != "column" or not doc.get("qualified_name"):
                continue
            qn = doc["qualified_name"]
            if qn in columns:
                continue
            if _domain_allowed(qn, question_lower, question_terms):
                columns.add(qn)
                sem_added += 1
            if sem_added >= 6:
                break

        # ── Always include name + district when listing / showing ────────────
        listing_verbs = {"show", "list", "display", "all", "find", "fetch", "get", "who"}
        if listing_verbs & question_terms:
            columns.update(["citizen.member_name", "citizen.district"])

        # ── Caste detail ──────────────────────────────────────────────────────
        if (KNOWN_CASTES & question_terms) or any(c in question_lower for c in KNOWN_CASTES):
            columns.add("citizen.caste")

        # ── Prune domain-irrelevant columns ──────────────────────────────────
        columns = _prune(columns, question_lower, question_terms)

        # Always at least one table
        tables = ["citizen"]
        confidence = (
            sum(d["score"] for d in docs[: min(5, len(docs))]) / max(1, min(5, len(docs)))
        )
        return RetrievalResult(
            question=question,
            tables=tables,
            columns=sorted(columns),
            relationships=[],
            documents=docs,
            confidence=round(confidence, 4),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _matches(term: str, question_lower: str, question_terms: set[str]) -> bool:
    norm = term.lower().strip()
    if not norm:
        return False
    return norm in question_lower if " " in norm else norm in question_terms


def _domain_allowed(qn: str, question_lower: str, question_terms: set[str]) -> bool:
    col = qn.split(".")[1]
    if col == "bank_name" and not (BANK_TERMS & question_terms):
        return False
    if col == "caste_category" and not (CASTE_CATEGORY_TERMS & question_terms):
        return False
    if col == "minority" and not (MINORITY_TERMS & question_terms):
        return False
    if col == "education" and not (EDUCATION_TERMS & question_terms):
        return False
    if col in {"bank_account", "ifsc_code"} and not (BANK_TERMS & question_terms):
        return False
    return True


def _prune(columns: set[str], question_lower: str, question_terms: set[str]) -> set[str]:
    pruned: set[str] = set()
    for qn in columns:
        col = qn.split(".")[1]
        if not _domain_allowed(qn, question_lower, question_terms):
            continue
        if col == "block" and not ({"block", "tehsil"} & question_terms or _has_location(question_lower)):
            continue
        if col == "gram_panchayat" and not ({"gram", "panchayat", "gp"} & question_terms):
            continue
        if col == "village" and not ("village" in question_terms or _has_location(question_lower)):
            continue
        if col == "ward" and "ward" not in question_terms:
            continue
        if col == "is_rural" and not (RURAL_TERMS & question_terms or "rural" in question_lower or "urban" in question_lower):
            continue
        if col in {"mobile_number", "jan_aadhaar_member_id", "enrollment_id"} and not (IDENTITY_TERMS & question_terms):
            continue
        pruned.add(qn)
    return pruned


def _has_location(question_lower: str) -> bool:
    for prep in LOCATION_PREPOSITIONS:
        m = re.search(rf"\b{prep}\s+([a-zA-Z][a-zA-Z-]*)\b", question_lower)
        if m and m.group(1).lower() not in LOCATION_STOPWORDS:
            return True
    return False
