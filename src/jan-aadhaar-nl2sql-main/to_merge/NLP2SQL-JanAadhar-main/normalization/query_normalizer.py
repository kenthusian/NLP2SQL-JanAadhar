from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41


COMMON_CANONICAL_TERMS = {
    "boy": ["boy", "boys", "male", "man", "men"],
    "girl": ["girl", "girls", "female", "woman", "women"],
    "beneficiary": ["beneficiary", "beneficiaries"],
    "pension": ["pension"],
    "ekyc": ["ekyc", "kyc", "e-kyc"],
    "aadhaar": ["aadhaar", "adhar", "aadhar"],
    "caste": ["caste", "cast"],
    "district": ["district", "zilla"],
}

# Bank abbreviation → full stored DB name mapping.
# Must be checked BEFORE the len < 4 guard in replace() because many abbreviations
# (sbi, pnb, bob, uco, cbi, ubi) are exactly 3 characters and would be skipped otherwise.
BANK_ABBREVIATIONS: dict[str, str] = {
    "sbi":   "STATE BANK OF INDIA",
    "pnb":   "PUNJAB NATIONAL BANK",
    "bob":   "BANK OF BARODA",
    "boi":   "BANK OF INDIA",
    "hdfc":  "HDFC BANK",
    "icici": "ICICI BANK LIMITED",
    "uco":   "UCO BANK",
    "cbi":   "CENTRAL BANK OF INDIA",
    "ubi":   "UNION BANK OF INDIA",
}

DIRECT_CORRECTIONS = {
    "femail": "female",
    "femal": "female",
    "benificiary": "beneficiary",
    "benificiaries": "beneficiaries",
    "beneficary": "beneficiary",
    "pensoin": "pension",
    "pention": "pension",
    "distict": "district",
    "distrct": "district",
    "jaipor": "Jaipur",
    "jaypur": "Jaipur",
    "jodpur": "Jodhpur",
    "bikanr": "Bikaner",
    "familys": "families",
    "famlies": "families",
}


@dataclass(frozen=True)
class QueryNormalizationResult:
    original: str
    normalized: str
    corrections: dict[str, str]


class QueryNormalizer:
    def __init__(self, threshold: int = 88):
        self.threshold = threshold
        self.accepted_terms = self._build_accepted_terms()
        self.correction_candidates = self._build_correction_candidates()

    def normalize(self, query: str) -> QueryNormalizationResult:
        corrections: dict[str, str] = {}
        
        # 1. Identify and protect proper nouns (following prepositions or in quotes)
        prepositions = {"in", "from", "at", "named", "called", "of"}
        words = re.findall(r"\b[a-zA-Z0-9]+\b", query)
        protected_words = set()
        
        for i in range(1, len(words)):
            if words[i-1].lower() in prepositions:
                protected_words.add(words[i].lower())
                
        # Protect words inside single/double quotes
        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", query):
            for word in re.findall(r"\b[a-zA-Z0-9]+\b", quoted):
                protected_words.add(word.lower())

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            lowered = token.lower()

            # ── MUST be FIRST: 3-char bank abbreviations (sbi, pnb, bob, uco…)
            #    would be silently skipped by the len < 4 guard further below.
            #    Bank terms are always expanded regardless of surrounding context.
            if lowered in BANK_ABBREVIATIONS:
                replacement = BANK_ABBREVIATIONS[lowered]
                corrections[token] = replacement
                return replacement

            # Protect proper nouns, unless they are explicitly in direct corrections
            is_protected = (lowered in protected_words)
            if is_protected and lowered in DIRECT_CORRECTIONS:
                is_protected = False
                
            if lowered in DIRECT_CORRECTIONS:
                replacement = DIRECT_CORRECTIONS[lowered]
                if replacement.lower() != lowered:
                    corrections[token] = replacement
                return replacement
                
            if lowered in self.accepted_terms:
                return token
                
            if len(lowered) < 4 or lowered.isdigit():
                return token
                
            candidate = process.extractOne(lowered, self.correction_candidates.keys(), scorer=fuzz.WRatio)
            if not candidate:
                return token
                
            matched, score, _ = candidate
            if score < self.threshold:
                return token
                
            # If the token is protected and it is not an extremely high confidence match, skip it
            if is_protected and score < 95:
                return token
                
            replacement = self.correction_candidates[matched]
            if replacement.lower() == lowered:
                return token
                
            corrections[token] = replacement
            return replacement

        normalized = re.sub(r"\b[a-zA-Z][a-zA-Z-]*\b", replace, query)
        return QueryNormalizationResult(original=query, normalized=normalized, corrections=corrections)

    def _build_accepted_terms(self) -> set[str]:
        accepted: set[str] = set()
        for district in RAJASTHAN_DISTRICTS_41:
            accepted.update(re.findall(r"[a-zA-Z]+", district.lower()))
            accepted.add(district.lower())
        for aliases in COMMON_CANONICAL_TERMS.values():
            for alias in aliases:
                accepted.add(alias.lower())
        for column in COLUMNS:
            accepted.add(column.column.replace("_", " ").lower())
            if column.semantic_name:
                accepted.add(column.semantic_name.lower())
            for alias in column.aliases:
                accepted.add(alias.lower())
        return accepted

    def _build_correction_candidates(self) -> dict[str, str]:
        candidates = {
            "female": "female",
            "male": "male",
            "beneficiary": "beneficiary",
            "beneficiaries": "beneficiaries",
            "pension": "pension",
            "ekyc": "ekyc",
            "aadhaar": "aadhaar",
            "caste": "caste",
            "district": "district",
        }
        candidates.update({district.lower(): district for district in RAJASTHAN_DISTRICTS_41})
        return candidates


def normalize_query(query: str) -> QueryNormalizationResult:
    return QueryNormalizer().normalize(query)
