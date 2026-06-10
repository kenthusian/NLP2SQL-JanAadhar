"""
ui/api.py — FastAPI backend for the Jan-Aadhaar NL2SQL pipeline.

Full pipeline (per request):
  Phase 2a → Normalize (RapidFuzz typo correction)
  Phase 2b → Domain dict (Regex SQL hints)
  Phase 2c → Semantic cache lookup (ChromaDB)
  Phase 3  → Schema RAG (ChromaDB)
  Phase 3  → Prompt assembly (system + schema + hints + few-shots)
  Phase 4  → LLM generation (Ollama qwen2.5-coder:3b)
  Phase 4  → AST validation + self-correction (sqlglot)
  Phase 4  → DuckDB execution
  Phase 4  → Cache store

Endpoints:
  POST /query        — full pipeline
  GET  /schema       — dataset column schema
  GET  /health       — DuckDB + ChromaDB + Ollama status
  GET  /cache/stats  — cache entry count
"""
from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COLUMN_SCHEMA, SCHEMA_JSON, TABLE_NAME
from db.query import execute_sql, health_check as duckdb_health
from llm.ollama_client import generate_sql, is_ollama_available
from llm.prompt_builder import build_correction_prompt, build_prompt, build_scaffold_prompt
from llm.sql_rewriter import rewrite as rewrite_sql
from logger import QueryTimer, get_logger
from rag.cache import cache_size, exact_lookup, semantic_lookup, store as cache_store
from rag.domain_dict import extract_sql_hints
from rag.normalizer import normalize_query
from rag.schema_index import all_column_names, build_index, retrieve as schema_retrieve
from validation.sql_guard import validate_with_retry
from llm.fast_sql import try_build_sql, build_partial_scaffold, _get_unmapped_words

log = get_logger("nl2sql.api")

L1_CACHE = TTLCache(maxsize=1000, ttl=3600)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Jan-Aadhaar NL2SQL API",
    description=(
        "Fully local Natural Language → SQL pipeline for the Jan-Aadhaar dataset. "
        "Zero external API calls. DuckDB + Ollama + ChromaDB."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    err = traceback.format_exc()
    log.error(f"UNHANDLED EXCEPTION: {err}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Startup: build/refresh schema index ──────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    import duckdb
    from config import DATA_DIR, TABLE_NAME
    log.info("Initializing DuckDB persistent connection...")
    try:
        con = duckdb.connect(":memory:")
        con.execute("PRAGMA threads = 8;")
        con.execute("PRAGMA memory_limit = '8GB';")
        parquet_glob = str(DATA_DIR / "**" / "*.parquet")
        con.execute(
            f"CREATE OR REPLACE VIEW {TABLE_NAME} AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"
        )
        app.state.duckdb_con = con
        log.info(f"DuckDB initialized successfully over {parquet_glob}.")
    except Exception as exc:
        log.error(f"DuckDB initialization failed: {exc}")

    log.info("Building / refreshing schema index...")
    try:
        await asyncio.to_thread(build_index, SCHEMA_JSON if SCHEMA_JSON.exists() else None)
        log.info("Schema index ready.")
    except Exception as exc:
        log.error(f"Schema index build failed: {exc}")


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    normalized_question: str
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    source: str                    # "cache" | "llm" | "deterministic" | "scaffold"
    domain_hints: list[str]        # regex hints injected into the prompt
    correction_attempts: int
    latency: dict[str, float]
    rag_columns: list[dict[str, str]] = []
    pipeline_steps: list[dict[str, Any]] = []  # step-by-step trace for UI


class HealthResponse(BaseModel):
    status: str
    duckdb: bool
    ollama: bool
    chromadb: bool
    cache_entries: int
    rows: int = 0
    schema_cols: int = 0


# ── Correction callback wired to prompt builder + ollama ─────────────────────
def _make_correction_fn(schema_cols, domain_hints):
    async def _fn(question: str, cols, failed_sql: str, error: str) -> tuple[str, float]:
        prompt = build_correction_prompt(question, cols, failed_sql, error)
        return await generate_sql(prompt)
    return _fn


# ── POST /query ───────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    latency: dict[str, float] = {}
    source = "llm"
    correction_attempts = 0
    domain_hints: list[str] = []
    pipeline_steps: list[dict[str, Any]] = []

    def add_step(name: str, status: str, ms: float, detail: str = ""):
        pipeline_steps.append({"name": name, "status": status, "ms": ms, "detail": detail})

    with QueryTimer(question) as timer:

        # ── Phase 2a: Normalize ───────────────────────────────────────────────
        t0 = time.perf_counter()
        normalized = normalize_query(question)
        norm_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency["normalize_ms"] = norm_ms
        add_step("Normalization", "used" if normalized != question else "skipped", norm_ms, f"Normalized to: {normalized}" if normalized != question else "")
        
        # ── Phase 2b: Domain dict ─────────────────────────────────────────────
        t1 = time.perf_counter()
        domain_hints, mapped_words = extract_sql_hints(normalized)
        unmapped_words = _get_unmapped_words(normalized, mapped_words)

        L1_key = normalized
        t_cache = time.perf_counter()
        if L1_key in L1_CACHE:
            cache_ms = round((time.perf_counter() - t_cache) * 1000, 2)
            latency["cache_lookup_ms"] = cache_ms
            log.info(f"[CACHE HIT - L1 RAM] in {latency['cache_lookup_ms']}ms")
            add_step("L1 Cache Lookup", "hit", cache_ms, "Exact match in RAM")
            
            cached = L1_CACHE[L1_key]
            latency["llm_generation_ms"] = 0.0
            latency["db_execution_ms"] = 0.0
            latency["total_ms"] = round((time.perf_counter() - timer._start) * 1000, 2)
            return QueryResponse(
                question=question,
                normalized_question=normalized,
                sql=cached["sql"],
                columns=cached["data"]["columns"],
                rows=cached["data"]["rows"],
                row_count=cached["data"]["row_count"],
                truncated=cached["data"]["truncated"],
                source="cache",
                domain_hints=domain_hints,
                correction_attempts=0,
                latency=latency,
                pipeline_steps=pipeline_steps,
            )

        add_step("L1 Cache Lookup", "miss", round((time.perf_counter() - t_cache) * 1000, 2))

        # ── Tier 2: Semantic Vector Cache (L2 - ChromaDB) ─────────────────────
        t_l2 = time.perf_counter()
        l2_result = await asyncio.to_thread(semantic_lookup, normalized, domain_hints, unmapped_words)
        cache_ms = round((time.perf_counter() - t_cache) * 1000, 2)
        latency["cache_lookup_ms"] = cache_ms
        
        if l2_result:
            cached_sql, cached_data_str = l2_result
            log.info(f"[CACHE HIT - L2 CHROMA] in {latency['cache_lookup_ms']}ms")
            add_step("L2 Semantic Cache", "hit", cache_ms, "Semantic match found in ChromaDB")
            
            cached_data = json.loads(cached_data_str) if cached_data_str and cached_data_str != "{}" else {}
            
            # Store it back in L1 for next time
            if cached_data:
                L1_CACHE[L1_key] = {"sql": cached_sql, "data": cached_data}
            
            latency["llm_generation_ms"] = 0.0
            latency["db_execution_ms"] = 0.0
            latency["total_ms"] = round((time.perf_counter() - timer._start) * 1000, 2)
            
            return QueryResponse(
                question=question,
                normalized_question=normalized,
                sql=cached_sql,
                columns=cached_data.get("columns", []),
                rows=cached_data.get("rows", []),
                row_count=cached_data.get("row_count", 0),
                truncated=cached_data.get("truncated", False),
                source="cache",
                domain_hints=domain_hints,
                correction_attempts=0,
                latency=latency,
                pipeline_steps=pipeline_steps,
            )

        add_step("L2 Semantic Cache", "miss", round((time.perf_counter() - t_l2) * 1000, 2))

        # ── Cache Miss: fire RAG + Deterministic in parallel ──────────────────
        # RAG takes ~400ms. Deterministic takes ~2ms. By firing both at the same
        # time, the RAG result is essentially free if deterministic fails — it's
        # already done or nearly done by the time we need it.
        t_llm_start = time.perf_counter()
        rag_task = asyncio.create_task(
            asyncio.to_thread(schema_retrieve, normalized)
        )
        fast_sql = try_build_sql(normalized, domain_hints, mapped_words)
        schema_cols = []

        if fast_sql:
            # ── MODE 1: Full Deterministic ────────────────────────────────────
            # Cancel the RAG task — we don't need it.
            rag_task.cancel()
            raw_sql = rewrite_sql(fast_sql, normalized)
            log.info(f"[DETERMINISTIC] Built SQL without LLM: {raw_sql[:80]!r}")
            det_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
            latency["llm_generation_ms"] = det_ms
            source = "deterministic"
            add_step("Deterministic Generation", "used", det_ms, "100% confidence, bypassed LLM")
            add_step("Schema RAG", "skipped", 0)
        else:
            add_step("Deterministic Generation", "miss", round((time.perf_counter() - t_llm_start) * 1000, 2), "Needed LLM fallback")
            
            # Await RAG — it was running in parallel, so this is nearly instant.
            t_rag = time.perf_counter()
            schema_cols = await rag_task
            rag_ms = round((time.perf_counter() - t_rag) * 1000, 2)
            latency["rag_ms"] = rag_ms
            add_step("Schema RAG", "used", rag_ms, f"Retrieved {len(schema_cols)} relevant columns")

            # Try to build a scaffold (Mode 2) — only possible if we have some domain hints
            scaffold = build_partial_scaffold(normalized, domain_hints, mapped_words)

            t_llm_gen = time.perf_counter()
            if scaffold and scaffold.unmapped_tokens:
                # ── MODE 2: Scaffold-Assisted ─────────────────────────────────
                # We know SOME conditions already — LLM fills in the rest.
                # Use a shorter, focused prompt with hard-wired WHERE constraints.
                prompt = build_scaffold_prompt(normalized, schema_cols, scaffold)
                log.info(
                    f"[SCAFFOLD] Pre-built WHERE: {scaffold.where_clause!r} | "
                    f"LLM resolves: {scaffold.unmapped_tokens}"
                )
                source = "scaffold"
            else:
                # ── MODE 3: Full LLM ──────────────────────────────────────────
                # No domain hints at all — full LLM with complete prompt.
                prompt = build_prompt(normalized, schema_cols, domain_hints)
                log.info("[FULL LLM] No scaffold possible, sending full prompt.")
                source = "llm"

            try:
                raw_sql, _ = await generate_sql(prompt)
            except ConnectionRefusedError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Ollama error: {exc}. Ensure Ollama is running and "
                           f"'qwen2.5-coder:3b' is pulled."
                )
            raw_sql = rewrite_sql(raw_sql, normalized)
            llm_ms = round((time.perf_counter() - t_llm_gen) * 1000, 2)
            latency["llm_generation_ms"] = llm_ms
            
            if source == "scaffold":
                add_step("Scaffold-Assisted LLM", "used", llm_ms, f"Pre-built WHERE + resolved {len(scaffold.unmapped_tokens)} tokens")
            else:
                add_step("Full LLM Generation", "used", llm_ms, "Full context sent to LLM")

        # ── Phase 4: AST validation + self-correction ─────────────────────
        t_val = time.perf_counter()
        allowed = all_column_names()
        _schema_cols = locals().get("schema_cols", [])
        result = await validate_with_retry(
            sql=raw_sql,
            allowed_columns=allowed,
            generate_fn=_make_correction_fn(_schema_cols, domain_hints),
            user_question=normalized,
            schema_cols=_schema_cols,
        )
        correction_attempts = result.correction_attempts

        val_ms = round((time.perf_counter() - t_val) * 1000, 2)
        latency["validation_ms"] = val_ms
        if correction_attempts > 0:
            add_step("AST Validation & Correction", "used", val_ms, f"{correction_attempts} corrections applied")
        else:
            add_step("AST Validation", "used", val_ms, "Passed on first try")

        if not result.ok:
            raise HTTPException(
                status_code=422,
                detail=f"SQL validation failed after {correction_attempts} "
                       f"correction attempt(s): {result.error}",
            )
        final_sql = result.sql

        # ── DuckDB execution ──────────────────────────────────────────────────
        t6 = time.perf_counter()
        try:
            db_result = await asyncio.to_thread(execute_sql, app.state.duckdb_con, final_sql)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"DuckDB error: {exc}")
        db_ms = round((time.perf_counter() - t6) * 1000, 2)
        latency["db_execution_ms"] = db_ms
        add_step("DuckDB Execution", "used", db_ms, f"Returned {db_result['row_count']} rows")

        # ── Cache store ───────────────────────────────────────────────────
        log.info(f"[{source.upper()} - {latency.get('llm_generation_ms', 0):.0f}ms LLM]")
        await asyncio.to_thread(cache_store, normalized, final_sql, db_result, domain_hints, unmapped_words)
        L1_CACHE[L1_key] = {"sql": final_sql, "data": db_result}

        latency["total_ms"] = round((time.perf_counter() - timer._start) * 1000, 2)

    return QueryResponse(
        question=question,
        normalized_question=normalized,
        sql=final_sql,
        columns=db_result["columns"],
        rows=db_result["rows"],
        row_count=db_result["row_count"],
        truncated=db_result["truncated"],
        source=source,
        domain_hints=domain_hints,
        correction_attempts=correction_attempts,
        latency=latency,
        rag_columns=schema_cols,
        pipeline_steps=pipeline_steps,
    )


# ── GET /schema ───────────────────────────────────────────────────────────────
@app.get("/schema")
async def schema_endpoint():
    import json
    if SCHEMA_JSON.exists():
        with open(SCHEMA_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {
        k: {"type": dtype, "description": desc}
        for k, (dtype, desc) in COLUMN_SCHEMA.items()
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    db_ok = await asyncio.to_thread(duckdb_health, app.state.duckdb_con)
    ollama_ok = is_ollama_available()
    try:
        entries = cache_size()
        chroma_ok = True
    except Exception:
        entries = 0
        chroma_ok = False

    row_count = 0
    try:
        row_count = app.state.duckdb_con.execute("SELECT COUNT(*) FROM aadhaar").fetchone()[0]
    except Exception:
        pass

    overall = "ok" if (db_ok and ollama_ok and chroma_ok) else "degraded"
    return HealthResponse(
        status=overall,
        duckdb=db_ok,
        ollama=ollama_ok,
        chromadb=chroma_ok,
        cache_entries=entries,
        rows=row_count,
        schema_cols=len(COLUMN_SCHEMA),
    )


# ── GET /cache/stats ──────────────────────────────────────────────────────────
@app.get("/cache/stats")
async def cache_stats():
    return {"cache_entries": cache_size()}
