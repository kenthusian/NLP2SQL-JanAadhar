"""
llm/ollama_client.py — HTTP client for the local Ollama inference server.

Calls POST /api/generate on localhost:11434 with streaming disabled
(we collect the full response in one shot since qwen2.5-coder:3b is fast
enough that streaming adds no UX benefit here).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LLM_TEMPERATURE, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL
from logger import get_logger

log = get_logger("nl2sql.ollama")

# Markdown fence stripper — handles ```sql … ``` or ``` … ```
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove markdown sql fences if present."""
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    
    # Strip any internal semicolons to prevent sqlglot parsing errors 
    # like "WHERE AGE > 50; LIMIT 500". The execution layer will add one at the end.
    text = text.replace(";", "")
    return text.strip()


async def generate_sql(prompt: str) -> tuple[str, float]:
    """
    Send *prompt* to Ollama and return the raw (cleaned) SQL string
    together with the inference latency in milliseconds.

    Args:
        prompt: Fully assembled LLM prompt from prompt_builder.

    Returns:
        (sql_string, latency_ms)

    Raises:
        httpx.HTTPError:       On network-level failure.
        ValueError:            If the response body is malformed.
        ConnectionRefusedError: If Ollama is not running.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",
        "format": "json",
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": 200,
            "stop": ["```", "\n\n", ";"],
        },
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise ConnectionRefusedError(
            f"Ollama unreachable at {OLLAMA_URL}. "
            "Start Ollama and ensure `qwen2.5-coder:3b` is pulled."
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    data = resp.json()
    raw_output: str = data.get("response", "")
    
    import json
    try:
        parsed_json = json.loads(raw_output)
        sql = parsed_json.get("query", "")
    except Exception:
        sql = raw_output
        
    sql = _strip_fences(sql)

    log.info(f"LLM inference {latency_ms}ms → {sql[:120]!r}")
    return sql, latency_ms


def is_ollama_available() -> bool:
    """Quick health-check — returns True if Ollama responds on /api/tags."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
