from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp
from dataclasses import dataclass, field

from database.schema_metadata import all_table_names, columns_by_table

# Dangerous AST nodes to block
_BLOCKED_NODE_TYPES = (
    exp.Drop, exp.Delete, exp.Update, exp.Insert, exp.Alter, 
    exp.Create, exp.Command, exp.TruncateTable
)

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

class SQLValidator:
    def __init__(self):
        self.tables = all_table_names()
        self.columns = columns_by_table()

    def validate(
        self,
        sql: str,
        allowed_tables: list[str] | None = None,
        allowed_columns: list[str] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        sql = sql.strip().rstrip(";")
        
        if not sql:
            return ValidationResult(False, ["Empty SQL statement."])

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            tree = sqlglot.parse_one(sql, dialect="sqlite")
        except sqlglot.errors.ParseError as exc:
            return ValidationResult(False, [f"SQL could not be parsed: {exc}"])
        except Exception as exc:
            return ValidationResult(False, [f"Unexpected parsing error: {exc}"])

        if tree is None:
            return ValidationResult(False, ["Empty or unparseable SQL"])

        # ── Root must be SELECT ───────────────────────────────────────────────
        if not isinstance(tree, exp.Select):
            errors.append(f"Root statement must be SELECT, got {type(tree).__name__}")

        # ── Block dangerous node types ────────────────────────────────────────
        for node in tree.walk():
            if isinstance(node, _BLOCKED_NODE_TYPES):
                errors.append(f"Blocked statement type: {type(node).__name__}")
                
        # ── Block JOINs (single-table enforcement) ────────────────────────────
        # Unless one of the tables is a results_ table (used for subset queries)
        has_results_table = False
        for table in tree.find_all(exp.Table):
            if table.name and "results_" in table.name.lower():
                has_results_table = True
                
        if not has_results_table and any(isinstance(node, exp.Join) for node in tree.walk()):
            errors.append("JOIN is not allowed — query only the citizen table.")

        # ── Check Tables ──────────────────────────────────────────────────────
        allowed_t = {t.lower() for t in (allowed_tables or self.tables)}
        referenced_tables = set()
        
        for table in tree.find_all(exp.Table):
            t_name = table.name.lower()
            if t_name:
                referenced_tables.add(t_name)
                if t_name not in allowed_t:
                    errors.append(f"Unknown/disallowed table: {t_name}")

        # ── Check Columns ─────────────────────────────────────────────────────
        # Build whitelist of allowed columns
        allowed_c = {c.lower() for c in (allowed_columns or [])}
        for t in (allowed_tables or self.tables):
            allowed_c.update(c.lower() for c in self.columns.get(t, []))
            
        # Extract aliases to allow them
        for alias_node in tree.find_all(exp.Alias):
            if alias_node.alias:
                allowed_c.add(alias_node.alias.lower())

        _ALWAYS_ALLOWED = {"*", "1", "current_date", "current_timestamp", "count"}
        
        for col_node in tree.find_all(exp.Column):
            col_name = col_node.name.lower() if col_node.name else ""
            if col_name and col_name not in allowed_c and col_name not in _ALWAYS_ALLOWED:
                errors.append(f"Unknown column referenced: {col_node.name}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

