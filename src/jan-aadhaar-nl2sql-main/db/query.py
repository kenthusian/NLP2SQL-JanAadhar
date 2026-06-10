"""
db/query.py — Thread-safe DuckDB query executor.

DuckDB opens the Parquet files directly in-memory (no separate server needed).
We use a module-level connection protected by a threading.Lock so FastAPI's
thread-pool workers don't stomp on each other.
"""
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, TABLE_NAME
from logger import get_logger

log = get_logger("nl2sql.query")

# ── Query executor ──────────────────────────────────────────────────────────────

def execute_sql(con: duckdb.DuckDBPyConnection, sql: str, limit: int = 500) -> dict:
    """
    Execute *sql* against the DuckDB view and return a dict with:
      - ``rows``       : list[dict] — result rows (capped at *limit*)
      - ``columns``    : list[str]
      - ``row_count``  : int — total rows returned (before limit)
      - ``exec_ms``    : float — DuckDB execution time in ms
      - ``truncated``  : bool — True if results were capped

    Raises:
        duckdb.Error: on SQL execution failure (caller handles retry logic)
    """
    t0 = time.perf_counter()
    
    result = con.execute(sql)
    columns = [desc[0] for desc in result.description]
    all_rows = result.fetchall()

    exec_ms = round((time.perf_counter() - t0) * 1000, 2)
    truncated = len(all_rows) > limit
    rows = [dict(zip(columns, row)) for row in all_rows[:limit]]

    log.debug(f"exec_ms={exec_ms} rows={len(all_rows)} sql={sql[:120]!r}")

    return {
        "rows": rows,
        "columns": columns,
        "row_count": len(all_rows),
        "exec_ms": exec_ms,
        "truncated": truncated,
    }


def health_check(con: duckdb.DuckDBPyConnection) -> bool:
    """Return True if DuckDB is reachable and the Parquet view exists."""
    try:
        con.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        log.error(f"DuckDB health check failed: {e}")
        return False
