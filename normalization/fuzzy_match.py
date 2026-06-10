from __future__ import annotations

import re
import pandas as pd
from rapidfuzz.distance import JaroWinkler

# Compile regex patterns for fuzzy name queries
_FUZZY_PATTERNS = [
    re.compile(r"\bsimilar\s+to\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bname(?:s)?\s+(?:is\s+)?like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bsound(?:s)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bspell(?:ed)?\s+like\s+([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bfuzzy\s+(?:search\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bapproximate\s+(?:matches\s+)?(?:for\s+)?([a-zA-Z\s]+)", re.IGNORECASE),
    re.compile(r"\bresembl(?:e|es|ing)\s+([a-zA-Z\s]+)", re.IGNORECASE),
]

# Words that indicate a stop in the extracted target name
_STOP_WORDS = {
    "in", "from", "at", "who", "where", "with", "and", "or",
    "whose", "of", "having", "is", "are", "limit", "show", "find"
}


def is_fuzzy_intent(question: str) -> bool:
    """
    Detects whether the question indicates a request for similar or fuzzy name matching.
    """
    for pattern in _FUZZY_PATTERNS:
        if pattern.search(question):
            return True
    return False


def extract_fuzzy_target(question: str) -> str | None:
    """
    Extracts the name to search for from a fuzzy query.
    Stops extracting if it encounters a stop word (e.g. location prepositions).
    """
    for pattern in _FUZZY_PATTERNS:
        match = pattern.search(question)
        if match:
            raw_target = match.group(1).strip()
            words = raw_target.split()
            name_words = []
            for word in words:
                if word.lower() in _STOP_WORDS:
                    break
                name_words.append(word)
            if name_words:
                return " ".join(name_words).strip().title()
    return None


def fuzzy_rerank(
    df: pd.DataFrame,
    target_name: str,
    threshold: float = 0.80,
    max_rows: int = 30
) -> pd.DataFrame:
    """
    Calculates Jaro-Winkler similarity scores between target_name and values in the
    first detected name column of the DataFrame. Filters by threshold, sorts descending,
    and returns up to max_rows.
    """
    if df.empty or not target_name:
        return df

    # Detect name column (single-table citizen schema)
    name_cols = ["member_name", "father_name", "mother_name", "spouse_name"]
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    match_col = None
    for col_key in name_cols:
        if col_key in df_cols_lower:
            match_col = df_cols_lower[col_key]
            break

    if not match_col:
        # Fallback to first column containing 'name'
        for col in df.columns:
            if "name" in col.lower():
                match_col = col
                break

    if not match_col:
        return df

    target_lower = target_name.lower()
    max_len_diff = 2 if len(target_name) <= 5 else 3
    scores = []
    for val in df[match_col]:
        if pd.isna(val) or not isinstance(val, str):
            scores.append(0.0)
        else:
            val_clean = val.strip()
            val_lower = val_clean.lower()
            words = [w.strip() for w in val_lower.split() if w.strip()]
            
            best_word_score = 0.0
            for word in words:
                len_diff = abs(len(word) - len(target_lower))
                is_prefix_match = len(target_lower) >= 5 and word.startswith(target_lower)
                if len_diff <= max_len_diff or is_prefix_match:
                    score = JaroWinkler.similarity(target_lower, word)
                    if score > 1.0:
                        score = score / 100.0
                    if score > best_word_score:
                        best_word_score = score
            scores.append(best_word_score)

    df_copy = df.copy()
    df_copy["similarity_score"] = scores
    df_copy = df_copy[df_copy["similarity_score"] >= threshold]
    df_copy = df_copy.sort_values(by="similarity_score", ascending=False)
    df_copy["similarity_score"] = df_copy["similarity_score"].round(2)
    return df_copy.head(max_rows)
