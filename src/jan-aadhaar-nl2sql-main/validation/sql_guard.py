"""
validation/sql_guard.py — AST-based SQL validation + self-correction loop.

Pipeline:
  1. sqlglot.parse_one(sql, dialect="duckdb")  — strict parse
  2. Assert root node is SELECT
  3. Walk AST — block any DDL / DML / destructive node types
  4. Extract all referenced columns — block any not in `allowed_columns`
  5. On failure → re-prompt Ollama (max MAX_CORRECTION_ATTEMPTS retries)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import sqlglot
import sqlglot.expressions as exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MAX_CORRECTION_ATTEMPTS
from logger import get_logger

log = get_logger("nl2sql.guard")

# ── Blocked AST node types ────────────────────────────────────────────────────
_BLOCKED_NODE_TYPES = (
    exp.Drop,
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.TruncateTable,
)


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    error: str = ""
    correction_attempts: int = 0


# ── Core validator ────────────────────────────────────────────────────────────

def validate(sql: str, allowed_columns: list[str]) -> ValidationResult:
    """
    Parse and validate *sql* against the allowed column whitelist.

    Args:
        sql:             The SQL string to validate.
        allowed_columns: List of column names from the schema (uppercase).

    Returns:
        ValidationResult(ok, sql, error)
    """
    sql = sql.strip().rstrip(";")

    # ── Parse ──────────────────────────────────────────────────────────────────
    try:
        tree = sqlglot.parse_one(sql, dialect="duckdb")
    except sqlglot.errors.ParseError as exc:
        return ValidationResult(ok=False, sql=sql, error=f"Parse error: {exc}")

    if tree is None:
        return ValidationResult(ok=False, sql=sql, error="Empty or unparseable SQL")

    # ── Root must be SELECT ────────────────────────────────────────────────────
    if not isinstance(tree, exp.Select):
        return ValidationResult(
            ok=False, sql=sql,
            error=f"Root statement must be SELECT, got {type(tree).__name__}"
        )

    # ── Block dangerous node types ─────────────────────────────────────────────
    for node in tree.walk():
        if isinstance(node, _BLOCKED_NODE_TYPES):
            return ValidationResult(
                ok=False, sql=sql,
                error=f"Blocked statement type: {type(node).__name__}"
            )

    # ── Column whitelist check ─────────────────────────────────────────────────
    allowed_upper = {c.upper() for c in allowed_columns}
    
    # Extract aliases to allow them in ORDER BY / GROUP BY
    for alias_node in tree.find_all(exp.Alias):
        if alias_node.alias:
            allowed_upper.add(alias_node.alias.upper())

    # Also allow common DuckDB pseudo-columns and aliases
    _ALWAYS_ALLOWED = {"*", "1", "CURRENT_DATE", "CURRENT_TIMESTAMP"}

    for col_node in tree.find_all(exp.Column):
        col_name = col_node.name.upper() if col_node.name else ""
        if col_name and col_name not in allowed_upper and col_name not in _ALWAYS_ALLOWED:
            return ValidationResult(
                ok=False, sql=sql,
                error=f"Unknown column referenced: {col_node.name!r}. "
                      f"Allowed: {sorted(allowed_upper)}"
            )

    log.info(f"SQL validated OK: {sql[:120]!r}")
    return ValidationResult(ok=True, sql=sql + ";")


# ── Self-correction loop ──────────────────────────────────────────────────────

async def validate_with_retry(
    sql: str,
    allowed_columns: list[str],
    generate_fn: Callable,
    user_question: str,
    schema_cols: list,
) -> ValidationResult:
    """
    Validate *sql*; on failure, invoke *generate_fn* to get a corrected version
    (up to MAX_CORRECTION_ATTEMPTS times).

    Args:
        sql:             Initial SQL from the LLM.
        allowed_columns: Column whitelist.
        generate_fn:     Callable(question, schema_cols, failed_sql, error) → (sql, ms)
                         This is wired to prompt_builder + ollama_client by the caller.
        user_question:   The original natural language question.
        schema_cols:     RAG-pruned column defs (passed through to generate_fn).

    Returns:
        ValidationResult — either a success or the last failure after all retries.
    """
    result = validate(sql, allowed_columns)
    attempts = 0

    while not result.ok and attempts < MAX_CORRECTION_ATTEMPTS:
        attempts += 1
        log.warning(
            f"Validation failed (attempt {attempts}/{MAX_CORRECTION_ATTEMPTS}): "
            f"{result.error}"
        )
        corrected_sql, _ = await generate_fn(user_question, schema_cols, sql, result.error)
        result = validate(corrected_sql, allowed_columns)
        result.correction_attempts = attempts

    if not result.ok:
        log.error(f"Validation failed after {attempts} correction attempts: {result.error}")

    return result
