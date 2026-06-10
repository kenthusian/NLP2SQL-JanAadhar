from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine


@dataclass(frozen=True)
class QueryResultPreview:
    rows: pd.DataFrame
    truncated: bool
    displayed_rows: int


def execute_select_preview(
    sql: str,
    max_rows: int = 200,
    database_url: str | None = None,
    fuzzy_target: str | None = None,
    is_fuzzy: bool = False,
    threshold: float = 0.80,
    cache_id: str | None = None,
) -> QueryResultPreview:
    from validation.sql_validator import SQLValidator
    validation = SQLValidator().validate(sql)
    if not validation.valid:
        raise ValueError(f"Refusing to execute unsafe SQL: {'; '.join(validation.errors)}")

    safe_sql = sql.strip().rstrip(";")
    fetch_limit = 1000 if is_fuzzy else max_rows + 1
    limited = f"SELECT * FROM ({safe_sql}) AS _q LIMIT {fetch_limit}"

    with get_engine(database_url).connect() as conn:
        result = conn.execute(text(limited))
        columns = list(result.keys())
        records = result.fetchall()

    db_truncated = len(records) > (1000 if is_fuzzy else max_rows)
    frame = pd.DataFrame(records[:( 1000 if is_fuzzy else max_rows)], columns=columns)

    if is_fuzzy and fuzzy_target:
        from normalization.fuzzy_match import fuzzy_rerank
        frame = fuzzy_rerank(frame, fuzzy_target, threshold=threshold, max_rows=max_rows)
        truncated = db_truncated
    else:
        truncated = len(records) > max_rows
        frame = frame.head(max_rows)

    # Persist result set to data cache for subset queries
    if cache_id:
        try:
            from caching.semantic_cache import SemanticCache
            SemanticCache().save_data_cache(cache_id, frame)
        except Exception:
            pass  # Never let caching break query display

    return QueryResultPreview(rows=frame, truncated=truncated, displayed_rows=len(frame))
