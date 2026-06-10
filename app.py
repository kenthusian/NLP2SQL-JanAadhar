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


# ── District lookups ─────────────────────────────────────────────────────────
_DISTRICT_LOWER: dict[str, str] = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}
_DISTRICTS_LOWER: set[str]       = {d.lower() for d in RAJASTHAN_DISTRICTS_41}
_DISTRICT_CANONICAL              = _DISTRICT_LOWER  # alias for clarity

# ── No-bank query signals ─────────────────────────────────────────────────────
_NO_BANK_WORDS = [
    "no bank", "without bank", "don't have", "do not have",
    "no account", "unbanked", "without account",
]

# ── Member-type explicit signals ──────────────────────────────────────────────
_MEMBER_TYPE_WORDS = [
    "regular member", "mem type", "member type", "hof", "head of family",
]

# ── Caste synonym groups (English + Hindi) ────────────────────────────────────
_CASTE_GROUPS = [
    {"rajput", "rajpoot", "\u0930\u093e\u091c\u092a\u0942\u0924"},
    {"jat", "\u091c\u093e\u091f"},
    {"mina", "meena", "\u092e\u0940\u0928\u093e"},
    {"brahman", "brahmin", "brahaman", "bhraman", "bharmn",
     "\u092c\u094d\u0930\u093e\u0939\u094d\u092e\u0923",
     "\u092c\u094d\u0930\u093e\u0939\u092e\u094d\u0923"},
    {"bairwa", "berwa", "\u092c\u0948\u0930\u0935\u093e"},
    {"gurjar", "gujar", "\u0917\u0941\u0930\u094d\u091c\u0930"},
    {"bazigar", "\u092c\u093e\u091c\u0940\u0917\u0930"},
    {"dhobi", "\u0927\u094b\u092c\u0940"},
    {"darzi", "\u0926\u0930\u094d\u091c\u0940"},
    {"fakir", "\u092b\u0915\u0940\u0930"},
    {"valmiki", "balmiki", "\u0935\u093e\u0932\u094d\u092e\u093f\u0915\u093f"},
    {"chhipa", "chhippa", "\u091b\u0940\u092a\u093e"},
    {"jain", "\u091c\u0948\u0928"},
    {"dangi", "\u0921\u093e\u0902\u0917\u0940"},
    {"agrawal", "agarwal", "\u0905\u0917\u094d\u0930\u0935\u093e\u0932"},
    {"mahajan", "\u092e\u0939\u093e\u091c\u0928"},
]

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

        # Post-processing: AST fixes + regex passes + unbanked guard
        sql = _post_process_sql(sql, fuzzy_target=fuzzy_target)
        sql = _fix_no_bank_sql(sql, question)

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
# Post-processor — AST core + targeted regex passes
# ─────────────────────────────────────────────────────────────────────────────
def _post_process_sql(sql: str, fuzzy_target: str | None = None) -> str:  # noqa: C901
    """Fix common LLM mistakes using sqlglot AST transforms + targeted regex."""
    import sqlglot
    import sqlglot.expressions as exp

    # ── AST pass ──────────────────────────────────────────────────────────────
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        tree = None

    if tree is not None:
        # 1. Remap legacy multi-table names → citizen
        for table in tree.find_all(exp.Table):
            if table.name.lower() in ("member", "family", "bank_details"):
                table.set("this", exp.Identifier(this="citizen"))

        # 2. Strip table-qualifier prefixes (e.g. citizen.age → age)
        for col in tree.find_all(exp.Column):
            if col.table:
                col.set("table", None)

        def _transform(node):
            if not isinstance(node, (exp.EQ, exp.Like)):
                return node
            left, right = node.left, node.right
            if not isinstance(left, exp.Column):
                return node
            col = left.name.lower()
            if not isinstance(right, exp.Literal) or not right.is_string:
                return node
            val = right.name.lower()
            orig = right.name

            # Gender
            if col == "gender":
                if val in ("male", "m", "boy", "boys", "man", "men", "gents",
                           "ladka", "ladke", "purusha"):
                    right.set("this", "Male")
                elif val in ("female", "f", "girl", "girls", "woman", "women",
                             "ladies", "lady", "ladki", "mahila", "aurat"):
                    right.set("this", "Female")

            # Caste category
            elif col == "caste_category":
                MAP = {
                    "general": "GEN", "gen": "GEN", "open": "GEN",
                    "unreserved": "GEN", "forward": "GEN", "ur": "GEN",
                    "obc": "OBC", "other backward": "OBC",
                    "sc": "SC", "dalit": "SC", "scheduled caste": "SC",
                    "st": "ST", "tribal": "ST", "adivasi": "ST",
                }
                if val in MAP:
                    right.set("this", MAP[val])

            # Marital status
            elif col == "marital_status":
                if val in ("widowed", "widower"):
                    right.set("this", "Widow")
                elif val in ("single", "bachelor", "spinster", "never married"):
                    right.set("this", "Unmarried")

            # is_rural
            elif col == "is_rural":
                if val in ("rural", "true", "yes", "1"):
                    return exp.EQ(this=left, expression=exp.Literal.number(1))
                elif val in ("urban", "false", "no", "0"):
                    return exp.EQ(this=left, expression=exp.Literal.number(0))

            # Education — illiterate is stored lowercase; others use LIKE
            elif col == "education":
                if val == "illiterate":
                    return exp.EQ(
                        this=exp.Lower(this=left),
                        expression=exp.Literal.string("illiterate"),
                    )
                # Any other exact = rewrite to LIKE
                stripped = orig.replace("%", "")
                return exp.Like(
                    this=left,
                    expression=exp.Literal.string(f"%{stripped}%"),
                )

            # bank_name → UPPER(col) LIKE '%UPPER_VAL%'
            elif col == "bank_name":
                val_stripped = orig.upper().replace("%", "")
                return exp.Like(
                    this=exp.Upper(this=left),
                    expression=exp.Literal.string(f"%{val_stripped}%"),
                )

            # District casing
            elif col == "district":
                canonical = _DISTRICT_CANONICAL.get(val)
                if canonical:
                    right.set("this", canonical)

            # Free-text columns → LIKE
            elif col in ("caste", "occupation", "member_name", "father_name",
                         "mother_name", "spouse_name"):
                if isinstance(node, exp.EQ):
                    stripped = orig.replace("%", "")
                    return exp.Like(
                        this=left,
                        expression=exp.Literal.string(f"%{stripped}%"),
                    )

            return node

        tree = tree.transform(_transform)
        sql = tree.sql(dialect="sqlite") + ";"

    # ── Regex pass 1: Caste bilingual group expansion ─────────────────────────
    def _caste_expand(match: re.Match) -> str:
        col = match.group(1)
        val = match.group(2).strip().lower()
        for group in _CASTE_GROUPS:
            if val in group:
                parts = [
                    f"{col} LIKE '%{t.title() if t.isascii() else t}%'"
                    for t in sorted(group, key=len, reverse=True)
                ]
                return "(" + " OR ".join(parts) + ")"
        return match.group(0)

    sql = re.sub(
        r"\b((?:\w+\.)?caste)\s+LIKE\s+'%?([^'%]+)%?'",
        _caste_expand, sql, flags=re.IGNORECASE,
    )

    # ── Regex pass 2: bank_account column alias normalisation ─────────────────
    sql = re.sub(r"\bbank_account_number\b", "bank_account", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bbank_account_no\b",     "bank_account", sql, flags=re.IGNORECASE)

    # ── Regex pass 3: bank_name LIKE without UPPER → add UPPER ───────────────
    def _bank_upper(m: re.Match) -> str:
        col, val = m.group(1), m.group(2).strip()
        if m.group(0).upper().startswith("UPPER("):
            return m.group(0)
        return f"UPPER({col}) LIKE '%{val.upper()}%'"

    sql = re.sub(
        r"\b((?:\w+\.)?bank_name)\s+LIKE\s+'%?([^'%]+)%?'",
        _bank_upper, sql, flags=re.IGNORECASE,
    )

    # ── Regex pass 4: district LIKE → = for known districts ──────────────────
    def _district_like(m: re.Match) -> str:
        col, val = m.group(1), m.group(2).strip()
        canonical = _DISTRICT_CANONICAL.get(val.lower())
        return f"{col} = '{canonical}'" if canonical else m.group(0)

    sql = re.sub(
        r"\b((?:\w+\.)?district)\s+LIKE\s+'%?([^'%]+?)%?'",
        _district_like, sql, flags=re.IGNORECASE,
    )

    # ── Regex pass 5: non-district location values → block/village redirect ───
    def _district_redirect(m: re.Match) -> str:
        prefix = m.group(1) or ""
        val = (m.group(2) or m.group(3) or "").strip()
        if not val or val.lower() in _DISTRICTS_LOWER:
            return m.group(0)
        return f"({prefix}block LIKE '%{val}%' OR {prefix}village LIKE '%{val}%')"

    sql = re.sub(
        r"\b((?:[A-Za-z_]\w*\.)?)"
        r"district\s*(?:=\s*'([^']+)'|LIKE\s*'%([^'%]+)%')",
        _district_redirect, sql, flags=re.IGNORECASE,
    )

    # ── Regex pass 6: COUNT(member.member_id) → COUNT(*) ─────────────────────
    sql = re.sub(
        r"COUNT\s*\(\s*\w+\.member_id\s*\)",
        "COUNT(*)", sql, flags=re.IGNORECASE,
    )

    # ── Regex pass 7: strip trailing commentary after first semicolon ─────────
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"

    # ── Regex pass 8: Fuzzy name broadening ──────────────────────────────────
    if fuzzy_target:
        prefix3 = fuzzy_target[:3]
        tgt_l = fuzzy_target.lower()

        def _fuzzy_repl(m: re.Match) -> str:
            col, val = m.group(1), m.group(2).strip()
            try:
                from rapidfuzz.distance import JaroWinkler
                score = JaroWinkler.similarity(tgt_l, val.lower())
            except Exception:
                score = 0.0
            if score >= 0.60 or val.lower() in tgt_l or tgt_l in val.lower():
                return f"{col} LIKE '%{prefix3}%'"
            return m.group(0)

        _NAME_COLS = "member_name|father_name|mother_name|spouse_name"
        sql = re.sub(
            rf"\b((?:\w+\.)?(?:{_NAME_COLS}))\s+LIKE\s+'%?([^'%]+)%?'",
            _fuzzy_repl, sql, flags=re.IGNORECASE,
        )

    return sql


# ─────────────────────────────────────────────────────────────────────────────
# Unbanked / no-account query fixer
# ─────────────────────────────────────────────────────────────────────────────
def _fix_no_bank_sql(sql: str, question: str) -> str:
    """Ensure LEFT JOIN + IS NULL for 'no bank account' queries."""
    if not any(w in question.lower() for w in _NO_BANK_WORDS):
        return sql

    # Inject bank_account IS NULL in WHERE if no bank-null check present
    if "bank_account" in sql.lower() and "IS NULL" not in sql.upper():
        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            sql = re.sub(
                r"\bWHERE\b",
                "WHERE bank_account IS NULL AND",
                sql, flags=re.IGNORECASE, count=1,
            )
        else:
            injected = re.sub(
                r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b",
                r"WHERE bank_account IS NULL \1",
                sql, flags=re.IGNORECASE, count=1,
            )
            sql = injected if injected != sql else sql.rstrip(";") + " WHERE bank_account IS NULL;"
    elif "bank_account" not in sql.lower():
        # LLM omitted bank_account entirely — add IS NULL guard
        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            sql = re.sub(
                r"\bWHERE\b",
                "WHERE bank_account IS NULL AND",
                sql, flags=re.IGNORECASE, count=1,
            )
        else:
            sql = sql.rstrip(";") + " WHERE bank_account IS NULL;"

    # Strip spurious member_type = 'MEM' the LLM sometimes adds
    if not any(w in question.lower() for w in _MEMBER_TYPE_WORDS):
        sql = re.sub(
            r"\s+AND\s+(?:\w+\.)?member_type\s*=\s*'MEM'",
            "", sql, flags=re.IGNORECASE,
        )

    return sql


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Jan Aadhaar NL2SQL — single-table edition")
    parser.add_argument("question",             nargs="*",           help="Natural language question")
    parser.add_argument("--build-index",        action="store_true", help="Rebuild FAISS schema index")
    parser.add_argument("--seed-demo-db",       action="store_true", help="Load dummy dataset into SQLite")
    parser.add_argument("--import-excel",                            help="Replace demo data with an Excel file")
    parser.add_argument("--show-results",       action="store_true", help="Print matching rows")
    parser.add_argument("--no-explain",         action="store_true", help="Skip EXPLAIN plan")
    parser.add_argument("--run-profile-query",  action="store_true", help="Profile query execution")
    args = parser.parse_args()

    if args.seed_demo_db:
        report = import_excel_dataset()
        print(f"Demo database ready — {report.rows_loaded} rows loaded from {report.source_name}")

    if args.import_excel:
        report = import_excel_dataset(args.import_excel)
        print(f"Imported {report.rows_loaded} records from {report.source_name}")

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

    if output.query_corrections:
        print("\nQuery spelling corrections")
        print(", ".join(f"{s} -> {t}" for s, t in output.query_corrections.items()))
        print(f"Normalized question: {output.normalized_question}")

    if output.validation_errors:
        print("\nValidation errors")
        print("; ".join(output.validation_errors))

    if args.show_results and output.sql:
        preview = execute_select_preview(
            output.sql, max_rows=20,
            fuzzy_target=output.fuzzy_target, is_fuzzy=output.is_fuzzy,
        )
        if output.is_fuzzy:
            print(f"\nSimilarity matches for '{output.fuzzy_target}' (Jaro-Winkler >= 0.80)")
        else:
            print("\nMatching entries")
        print(preview.rows.to_string(index=False) if not preview.rows.empty else "No matching entries.")
        if preview.truncated:
            print("Showing the first 20 rows only.")

    if output.optimization:
        print("\nExecution plan")
        print("\n".join(output.optimization.execution_plan))
        print(f"\nPlanning/explain time: {output.optimization.execution_time_ms} ms")
        if output.optimization.index_recommendations:
            print("\nIndex recommendations")
            print("\n".join(output.optimization.index_recommendations))


if __name__ == "__main__":
    run_cli()
