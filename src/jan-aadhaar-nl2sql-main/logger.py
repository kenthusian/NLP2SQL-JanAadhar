"""
logger.py — Structured JSON logger for the NL2SQL pipeline.
Every query is logged as a single JSON line with timing breakdown.
"""
import logging
import json
import time
from pathlib import Path
from config import LOG_FILE

# Ensure the log directory exists
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Root logger setup ─────────────────────────────────────────────────────────
_fmt = logging.Formatter("%(message)s")

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

_root = logging.getLogger("nl2sql")
_root.setLevel(logging.DEBUG)
_root.addHandler(_file_handler)
_root.addHandler(_console_handler)


def get_logger(name: str = "nl2sql") -> logging.Logger:
    """Return a child logger."""
    return logging.getLogger(name)


class QueryTimer:
    """
    Context-manager timer that produces a structured log entry on exit.

    Usage::

        with QueryTimer("my_query") as t:
            t.mark("cache_lookup")
            ... do cache lookup ...
            t.mark("llm_inference")
            ... call LLM ...
            t.mark("duckdb_exec")
            ... run DuckDB ...
        # On exit, a JSON line is written to the log file.
    """

    def __init__(self, prompt: str):
        self.prompt = prompt
        self._marks: list[tuple[str, float]] = []
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._last = self._start
        return self

    def mark(self, label: str):
        now = time.perf_counter()
        self._marks.append((label, round((now - self._last) * 1000, 2)))
        self._last = now

    def __exit__(self, *_):
        total_ms = round((time.perf_counter() - self._start) * 1000, 2)
        record = {
            "prompt": self.prompt,
            "total_ms": total_ms,
            "stages": {label: ms for label, ms in self._marks},
        }
        _root.info(json.dumps(record, ensure_ascii=False))


def log_event(event: str, **kwargs):
    """Log a one-off structured event (non-query)."""
    record = {"event": event, **kwargs}
    _root.info(json.dumps(record, ensure_ascii=False))
