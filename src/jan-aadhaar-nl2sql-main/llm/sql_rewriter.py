"""
llm/sql_rewriter.py — Deterministic post-generation SQL rewriter.

Fixes common LLM quantifier mis-translations BEFORE the SQL hits
the cache or DuckDB, so wrong results are never stored.

Currently handles:
  - HAVING COUNT(*) <op> <n> operator AND number correction based on English quantifiers
    in the original user question ("at least 2" → >= 2, "exactly 3" → = 3, etc.)
  - IS_RURAL value normalisation ('R'/'U' → '1'/'0' and vice-versa)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger import get_logger

log = get_logger("nl2sql.rewriter")


# ── Quantifier → SQL operator mapping ────────────────────────────────────────
# Each entry: (regex_pattern_in_question, correct_sql_op)
# Patterns capture the number that follows the quantifier word.
# ORDER MATTERS: more specific patterns first.
_QUANTIFIER_RULES: list[tuple[re.Pattern, str]] = [
    # "at least N" / "minimum N" / "no fewer than N" / "no less than N" → >=
    # NOTE: 'no fewer than' before 'fewer than', 'no less than' before 'less than'
    (re.compile(r"\b(?:at least|minimum|min|no fewer than|no less than)\s+(\d+)"), ">="),
    # "no more than N" / "at most N" / "maximum N" / "up to N" → <=
    # NOTE: 'no more than' before 'more than'
    (re.compile(r"\b(?:no more than|at most|maximum|max|up to)\s+(\d+)"), "<="),
    # "more than N" / "greater than N" / "above N" / "over N" → >
    (re.compile(r"\b(?:more than|greater than|above|over|exceeding|exceeds)\s+(\d+)"), ">"),
    # "exactly N" / "just N" / "precisely N" / "only N" → =
    (re.compile(r"\b(?:exactly|just|precisely|only)\s+(\d+)"), "="),
    # "fewer than N" / "less than N" / "under N" / "below N" → <
    (re.compile(r"\b(?:fewer than|less than|under|below)\s+(\d+)"), "<"),
]

# Pattern to find HAVING COUNT(...) <op> <number> in generated SQL
_HAVING_RE = re.compile(
    r"(HAVING\s+COUNT\([^)]*\)\s*)([><=!]+)(\s*\d+)",
    re.IGNORECASE,
)


def _intended_op_and_number(question: str) -> tuple[str, str] | None:
    """
    Return (sql_operator, number_str) if the question contains a clear quantifier,
    else None.
    """
    q = question.lower()
    for pattern, op in _QUANTIFIER_RULES:
        m = pattern.search(q)
        if m:
            return op, m.group(1)
    return None


def rewrite_having_operator(sql: str, question: str) -> str:
    """
    If the user question contains a clear quantifier phrase AND the generated SQL
    contains a HAVING COUNT clause, correct both the comparison operator AND the
    number to match intent.
    Returns the (possibly corrected) SQL string.
    """
    result = _intended_op_and_number(question)
    if result is None:
        return sql  # ambiguous — leave as-is

    intended_op, intended_num = result

    def _replacer(m: re.Match) -> str:
        current_op = m.group(2).strip()
        current_num = m.group(3).strip()
        if current_op == intended_op and current_num == intended_num:
            return m.group(0)  # already correct
        corrected = m.group(1) + intended_op + " " + intended_num
        log.info(
            f"[REWRITER] HAVING corrected: '{current_op} {current_num}' → "
            f"'{intended_op} {intended_num}' (question: {question!r})"
        )
        return corrected

    return _HAVING_RE.sub(_replacer, sql)


def rewrite_is_rural(sql: str) -> str:
    """
    Normalise IS_RURAL values: the dataset stores '1' (rural) and '0' (urban).
    Fix any LLM that uses 'R'/'U' or 'Rural'/'Urban' literals.
    """
    sql = re.sub(r"IS_RURAL\s*=\s*'[Rr]'", "IS_RURAL = '1'", sql)
    sql = re.sub(r"IS_RURAL\s*=\s*'[Uu]'", "IS_RURAL = '0'", sql)
    sql = re.sub(r"IS_RURAL\s*=\s*'[Rr]ural'", "IS_RURAL = '1'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"IS_RURAL\s*=\s*'[Uu]rban'", "IS_RURAL = '0'", sql, flags=re.IGNORECASE)
    return sql


def rewrite(sql: str, question: str) -> str:
    """
    Apply all deterministic rewrites in sequence.
    Returns the corrected SQL string.
    """
    sql = rewrite_having_operator(sql, question)
    sql = rewrite_is_rural(sql)
    return sql
