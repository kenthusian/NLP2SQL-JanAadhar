"""
app.py — Jan Aadhaar NL2SQL pipeline (single-table, with LLM bypass)

Pipeline order (sequential short-circuit):
  1. Semantic Cache lookup   — ~10 ms
  2. Procedural LLM Bypass   — ~5  ms  (simple ≤3-clause queries)
  3. LLM generation          — ~2-8 s  (fallback for complex queries)
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from config.settings import settings
from database.excel_importer import import_excel_dataset
from database.query_results import execute_select_preview
from database.schema_metadata import RAJASTHAN_DISTRICTS_41
from embeddings.faiss_store import FaissSchemaStore
from llm.ollama_client import OllamaModelManager, OllamaSqlGenerator, _clean_sql
from normalization.query_normalizer import normalize_query
from optimization.query_optimizer import OptimizationReport, QueryOptimizer
from prompting.prompt_builder import PromptBuilder
from retrieval.schema_retriever import SchemaRetriever
from validation.sql_validator import SQLValidator
from caching.semantic_cache import SemanticCache


# ── District lookup ───────────────────────────────────────────────────────────
_DISTRICT_LOWER: dict[str, str] = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}

# ── Module-level cache singleton (loads FAISS index once per process) ─────────
_cache_instance: SemanticCache | None = None


def _get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


# ─────────────────────────────────────────────────────────────────────────────
# PipelineOutput
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineOutput:
    question: str
    normalized_question: str
    query_corrections: dict[str, str]
    sql: str
    retrieved_tables: list[str]
    retrieved_columns: list[str]
    confidence: float
    validation_errors: list[str]
    optimization: OptimizationReport | None
    is_fuzzy: bool = False
    fuzzy_target: str | None = None
    cache_id: str | None = None
    source: str = "llm"          # "cache" | "bypass" | "llm"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def generate_sql_pipeline(
    question: str,
    ask_model_pull: bool = True,
    include_optimization: bool = True,
    run_query_for_profile: bool = False,
    stream_callback: callable = None,
) -> PipelineOutput:

    normalized = normalize_query(question)

    # Fuzzy intent detection (done early — bypass should honour it)
    from normalization.fuzzy_match import is_fuzzy_intent, extract_fuzzy_target
    is_fuzzy = is_fuzzy_intent(question)
    fuzzy_target: str | None = None
    if is_fuzzy:
        t = extract_fuzzy_target(question)
        fuzzy_target = t if (t and len(t) >= 3) else None
        if not fuzzy_target:
            is_fuzzy = False

    # ── STEP 0: Fast Path ─────────────────────────────────────────────────────
    from llm.fast_path import FastPathEngine
    fast_engine = FastPathEngine()
    fast_sql = fast_engine.generate_sql_fast(question)
    
    if fast_sql:
        opt = QueryOptimizer().profile(fast_sql, run_query=run_query_for_profile) if include_optimization else None
        return PipelineOutput(
            question=question,
            normalized_question=normalized.normalized,
            query_corrections=normalized.corrections,
            sql=fast_sql,
            retrieved_tables=["citizen"],
            retrieved_columns=["(fast_path)"],
            confidence=1.0,
            validation_errors=[],
            optimization=opt,
            is_fuzzy=is_fuzzy,
            fuzzy_target=fuzzy_target,
            cache_id=None,
            source="fast_path",
        )

    # ── STEP 1: Semantic Cache ────────────────────────────────────────────────
    # Singleton: reuse FAISS index and SQLite connection across calls
    cache = _get_cache()
    cached_entry = cache.search(normalized.normalized)
    is_exact_match = False
    generator_for_equiv: OllamaSqlGenerator | None = None

    if cached_entry:
        if cached_entry.similarity >= cache.exact_match_threshold:
            is_exact_match = True
        elif cached_entry.similarity >= cache.template_match_threshold:
            # Attempt 0ms AST Parameter Swapping FIRST
            swapped_sql = fast_engine.swap_ast_parameters(
                cached_entry.sql, cached_entry.original_question, question
            )
            if swapped_sql:
                opt = QueryOptimizer().profile(swapped_sql, run_query=run_query_for_profile) if include_optimization else None
                return PipelineOutput(
                    question=question,
                    normalized_question=normalized.normalized,
                    query_corrections=normalized.corrections,
                    sql=swapped_sql,
                    retrieved_tables=["citizen"],
                    retrieved_columns=["(cached_swapped)"],
                    confidence=cached_entry.similarity,
                    validation_errors=[],
                    optimization=opt,
                    is_fuzzy=is_fuzzy,
                    fuzzy_target=fuzzy_target,
                    cache_id=cached_entry.id,
                    source="cache_swapped",
                )

            # Lightweight LLM classification: 3-token budget, very fast
            generator_for_equiv = OllamaSqlGenerator()
            resp = generator_for_equiv.generate(
                f"Compare these two database queries:\n"
                f"Q1: '{cached_entry.original_question}'\n"
                f"Q2: '{question}'\n"
                f"Do they request EXACTLY the same data with EXACTLY the same filters? "
                f"If Q2 has an extra word like 'jaat', 'male', 'Jaipur' that filters the data more than Q1, they are NOT the same.\n"
                f"Answer YES or NO only.",
                stream=False, max_tokens=3,
            ).strip().upper()
            if "YES" in resp:
                is_exact_match = True

    if is_exact_match:
        opt = QueryOptimizer().profile(cached_entry.sql, run_query=run_query_for_profile) if include_optimization else None
        return PipelineOutput(
            question=question,
            normalized_question=normalized.normalized,
            query_corrections=normalized.corrections,
            sql=cached_entry.sql,
            retrieved_tables=["citizen"],
            retrieved_columns=["(cached)"],
            confidence=cached_entry.similarity,
            validation_errors=[],
            optimization=opt,
            is_fuzzy=is_fuzzy,
            fuzzy_target=fuzzy_target,
            cache_id=cached_entry.id,
            source="cache",
        )



    # ── STEP 2: LLM Generation ────────────────────────────────────────────────
    manager = OllamaModelManager()
    manager.ensure_model(settings.sql_model, ask_permission=ask_model_pull)
    manager.ensure_model(settings.embedding_model, ask_permission=ask_model_pull)

    store = FaissSchemaStore()
    store.build()
    retrieval = SchemaRetriever(store).retrieve(normalized.normalized)
    prompt_builder = PromptBuilder()
    generator = generator_for_equiv or OllamaSqlGenerator()
    validator = SQLValidator()

    previous_error: str | None = None
    sql = ""
    validation_errors: list[str] = []
    final_sql_is_valid = False
    cache_id: str | None = None

    for attempt in range(settings.max_retries):
        # If there's a similar cached entry use it as a fast-edit base
        if cached_entry and cached_entry.similarity >= cache.template_match_threshold and attempt == 0:
            has_data = cache.has_data_cache(cached_entry.id)
            if has_data:
                tbl = f"results_{cached_entry.id.replace('-', '_')}"
                prompt = (
                    f"The user previously asked '{cached_entry.original_question}' and results are in table `{tbl}`.\n"
                    f"The user now asks: '{question}'.\n"
                    f"If the new question is a subset of the cached data, query `{tbl}` directly.\n"
                    f"Otherwise modify this SQL for the citizen table: {cached_entry.sql}\n"
                    "Output ONLY the SQL."
                )
            else:
                prompt = (
                    f"Previous similar question: '{cached_entry.original_question}'\n"
                    f"Its SQL: {cached_entry.sql}\n"
                    f"New question: '{question}'\n"
                    "Modify the SQL minimally to answer the new question. Output ONLY the SQL."
                )
        else:
            prompt = prompt_builder.build(retrieval, previous_error=previous_error)

        # Stream first attempt if callback provided
        if stream_callback and attempt == 0:
            chunks: list[str] = []
            for chunk in generator.generate(prompt, stream=True):
                chunks.append(chunk)
                stream_callback(chunk)
            sql = _clean_sql("".join(chunks))
        else:
            sql = generator.generate(prompt, stream=False)

        # Light post-processing for the single-table world
        sql = _post_process_sql(sql)

        validation = validator.validate(
            sql,
            allowed_tables=["citizen"] + (
                [f"results_{cached_entry.id.replace('-', '_')}"] if cached_entry and cache.has_data_cache(cached_entry.id) else []
            ),
        )
        validation_errors = validation.errors

        if validation.valid:
            # Execution trial — catch SQLite runtime errors early
            from database.connection import get_engine
            from sqlalchemy import text as _text
            try:
                with get_engine().connect() as conn:
                    conn.execute(_text(f"SELECT * FROM ({sql.rstrip(';')}) LIMIT 0"))
                final_sql_is_valid = True
                break
            except Exception as db_exc:
                err = str(db_exc)
                validation_errors.append(f"Database error: {err}")
                previous_error = f"The SQL caused a database error: {err}. Please fix it."
        else:
            previous_error = "; ".join(validation_errors)

    if final_sql_is_valid:
        cache_id = cache.add(question, normalized.normalized, sql)

    opt = QueryOptimizer().profile(sql, run_query=run_query_for_profile) if (include_optimization and final_sql_is_valid) else None

    return PipelineOutput(
        question=question,
        normalized_question=normalized.normalized,
        query_corrections=normalized.corrections,
        sql=sql if final_sql_is_valid else "",
        retrieved_tables=retrieval.tables,
        retrieved_columns=retrieval.columns,
        confidence=retrieval.confidence,
        validation_errors=validation_errors,
        optimization=opt,
        is_fuzzy=is_fuzzy,
        fuzzy_target=fuzzy_target,
        cache_id=cache_id,
        source="llm",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight post-processor for single-table SQL
# ─────────────────────────────────────────────────────────────────────────────
def _post_process_sql(sql: str) -> str:
    """Fix the most common LLM mistakes for the single citizen table using AST transformation."""
    import sqlglot
    import sqlglot.expressions as exp

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        # If parsing fails, just return cleaned string as fallback
        return sql.strip().rstrip(";") + ";"

    # 1. Ensure table is citizen
    for table in tree.find_all(exp.Table):
        if table.name.lower() in ("member", "family", "bank_details"):
            table.set("this", exp.Identifier(this="citizen"))

    # 2. Remove table qualifiers (e.g., citizen.age -> age)
    for col in tree.find_all(exp.Column):
        if col.table:
            col.set("table", None)
            
    # 3. Handle specific conditions programmatically
    def transform_condition(node):
        if not isinstance(node, (exp.EQ, exp.Like)):
            return node
            
        left = node.left
        right = node.right
        
        if not isinstance(left, exp.Column):
            return node
            
        col_name = left.name.lower()
        
        # Only process string literals on the right hand side
        if not isinstance(right, exp.Literal) or not right.is_string:
            return node
            
        val = right.name.lower()
        original_val = right.name
        
        # Gender normalization
        if col_name == "gender":
            if val in ("male", "m", "boy", "boys", "man", "men"):
                right.set("this", "Male")
            elif val in ("female", "f", "girl", "girls", "woman", "women"):
                right.set("this", "Female")
                
        # Caste category normalization
        elif col_name == "caste_category":
            if val == "general": right.set("this", "GEN")
            elif val == "obc": right.set("this", "OBC")
            elif val == "sc": right.set("this", "SC")
            elif val == "st": right.set("this", "ST")
            
        # Marital status
        elif col_name == "marital_status":
            if val == "widowed": right.set("this", "Widow")
            
        # is_rural boolean
        elif col_name == "is_rural":
            if val in ("rural", "true", "yes", "1"):
                return exp.EQ(this=left, expression=exp.Literal.number(1))
            elif val in ("urban", "false", "no", "0"):
                return exp.EQ(this=left, expression=exp.Literal.number(0))
                
        # Education illiterate
        elif col_name == "education" and val == "illiterate":
            new_left = exp.Lower(this=left)
            right.set("this", "illiterate")
            return exp.EQ(this=new_left, expression=right)
            
        # bank_name UPPER LIKE
        elif col_name == "bank_name":
            new_left = exp.Upper(this=left)
            val_stripped = original_val.upper().replace("%", "")
            new_right = exp.Literal.string(f"%{val_stripped}%")
            return exp.Like(this=new_left, expression=new_right)
            
        # General LIKE fields (fuzzy match)
        elif col_name in ("caste", "member_name", "father_name", "mother_name", "spouse_name"):
            if isinstance(node, exp.EQ):
                val_stripped = original_val.replace("%", "")
                new_right = exp.Literal.string(f"%{val_stripped}%")
                return exp.Like(this=left, expression=new_right)
                
        return node

    # Transform all nodes
    tree = tree.transform(transform_condition)

    return tree.sql(dialect="sqlite") + ";"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Jan Aadhaar NL2SQL — single-table edition")
    parser.add_argument("question", nargs="*", help="Natural language question")
    parser.add_argument("--build-index",       action="store_true", help="Rebuild FAISS schema index")
    parser.add_argument("--seed-demo-db",      action="store_true", help="Load dummy dataset into SQLite")
    parser.add_argument("--show-results",      action="store_true", help="Print matching rows")
    parser.add_argument("--no-explain",        action="store_true", help="Skip EXPLAIN plan")
    parser.add_argument("--run-profile-query", action="store_true", help="Profile query execution")
    args = parser.parse_args()

    if args.seed_demo_db:
        report = import_excel_dataset()
        print(f"Demo database ready — {report.rows_loaded} rows loaded from {report.source_name}")

    manager = OllamaModelManager()
    manager.ensure_model(settings.sql_model)
    manager.ensure_model(settings.embedding_model)

    if args.build_index:
        FaissSchemaStore().build(force=True)
        print(f"FAISS index rebuilt at {settings.faiss_index_path}")

    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask a Jan Aadhaar database question: ").strip()

    output = generate_sql_pipeline(
        question,
        ask_model_pull=False,
        include_optimization=not args.no_explain,
        run_query_for_profile=args.run_profile_query,
    )

    print(f"\n[Source: {output.source.upper()}]")
    print("\nGenerated SQL")
    print(output.sql)
    print(f"\nConfidence: {output.confidence}")
    if output.validation_errors:
        print("\nValidation errors")
        print("; ".join(output.validation_errors))
    if args.show_results and output.sql:
        preview = execute_select_preview(
            output.sql, max_rows=20,
            fuzzy_target=output.fuzzy_target, is_fuzzy=output.is_fuzzy,
        )
        print("\nMatching entries")
        print(preview.rows.to_string(index=False) if not preview.rows.empty else "No matching entries.")
    if output.optimization:
        print("\nExecution plan")
        print("\n".join(output.optimization.execution_plan))


if __name__ == "__main__":
    run_cli()
