"""
llm/prompt_builder.py — Assemble the structured prompt for Qwen2.5-Coder.

The prompt has four blocks:
  1. SYSTEM  — dialect enforcement + output format instructions
  2. SCHEMA  — RAG-pruned column definitions
  3. FEW-SHOT — 5 realistic Jan-Aadhaar DuckDB examples
  4. QUESTION — the user's natural language question

The LLM is instructed to emit ONLY raw SQL — no markdown, no explanations.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TABLE_NAME

if TYPE_CHECKING:
    from rag.schema_index import ColumnDef

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = f"""\
You are an expert DuckDB SQL query generator for a government welfare dataset.

STRICT RULES:
- Output ONLY the raw JSON object — no markdown, no code fences, no explanation.
- The output format must be strictly exactly like this: {{"query": "SELECT ..."}}
- Use DuckDB SQL syntax only (e.g. STRPTIME, DATE_DIFF, TRY_CAST, REGEXP_MATCHES, ILIKE).
- The default table name is `{TABLE_NAME}`.
- IF AND ONLY IF the user asks for high-level aggregate counts or metrics (e.g. "Total families by district", "How many males in rural areas"), query the pre-aggregated rollup file directly: `read_parquet('data/district_rollup.parquet')`. This rollup has columns: DISTRICT_NAME_ENG, IS_RURAL, GENDER, CASTE_CATEGORY, total_members.
- Only reference columns that appear exactly as written in the schema below. NEVER hallucinate column names.
- If the user specifies a value (like a specific caste or district), put it in the WHERE clause as a string (e.g. CASTE ILIKE '%jaat%'). DO NOT append the value to the column name (e.g. NEVER output CASTE_JAAT).
- Always use SELECT — never emit INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER.
- String comparisons should be case-insensitive; prefer ILIKE or LOWER().
- For date arithmetic use CURRENT_DATE - INTERVAL 21 YEAR or DATE_DIFF. NEVER use DATE_SUB() or DATE_ADD(). Do not use AGE as a function (AGE is a numeric column in the schema).
- For unmapped surnames or communities (e.g. "sharmas", "saini"), strip any plural 's' and search BOTH the NAME_EN and CASTE columns (e.g. `(NAME_EN ILIKE '%sharma%' OR CASTE ILIKE '%sharma%')`). Only do this if a domain hint is not provided.
- Add LIMIT 500 unless the query is an aggregation (COUNT, SUM, AVG, etc.). DO NOT include any semicolons (;) anywhere in the output.
- HOUSEHOLD COUNTING RULE: When explicitly asked to COUNT or find "how many" households/families meeting a grouped condition, ALWAYS use a subquery:
  `SELECT COUNT(*) AS household_count FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE ... GROUP BY ENROLLMENT_ID HAVING COUNT(*) OP N)`
  NEVER use `SELECT COUNT(DISTINCT ENROLLMENT_ID) ... GROUP BY ENROLLMENT_ID`.
- HOUSEHOLD LISTING RULE: When explicitly asked to LIST or SHOW households/families meeting a grouped condition (e.g. "show list of all households..."), do NOT use a subquery or COUNT(*). Instead, list the ENROLLMENT_ID directly:
  `SELECT ENROLLMENT_ID FROM aadhaar WHERE ... GROUP BY ENROLLMENT_ID HAVING COUNT(*) OP N LIMIT 500`
- FAMILY SIZE RULE: "show families with more than N members" or "which family has the most members" means:
  `SELECT ENROLLMENT_ID, COUNT(*) AS member_count FROM aadhaar GROUP BY ENROLLMENT_ID HAVING COUNT(*) > N ORDER BY member_count DESC LIMIT 500`.
  Do NOT use a subquery here — we want to LIST the families, not count them.
- MULTI-LOCATION RULE: If the user asks for people from TWO OR MORE places (e.g. "Srinagar and Beejasar", "Jaipur or Kota"), use OR in the WHERE clause across all relevant location columns (VILL_NAME_ENG, GP_NAME_ENG, DISTRICT_NAME_ENG etc.): `(VILL_NAME_ENG ILIKE '%Srinagar%' OR VILL_NAME_ENG ILIKE '%Beejasar%')`.
- AMBIGUOUS LOCATION RULE: When the user says 'from X', 'in X', or 'of X' without specifying whether X is a village, district, block, or GP, ALWAYS search across ALL location columns with OR: `(VILL_NAME_ENG ILIKE '%X%' OR GP_NAME_ENG ILIKE '%X%' OR BLOCK_NAME_ENG ILIKE '%X%' OR DISTRICT_NAME_ENG ILIKE '%X%')`. Never assume a single column. Only use a single column when the user explicitly says 'from village X', 'from district X', 'from block X', etc.
- RELATIONAL QUERY RULE: To find members based on traits of their family/parents (e.g., "sons of farmers", "wives of men in Jaipur"), you MUST use an IN subquery on ENROLLMENT_ID. DO NOT USE JOINs or SELF-JOINs. ALWAYS select from a single 'aadhaar' table. Example: `WHERE RELATION_WITH_HOF ILIKE '%Son%' AND ENROLLMENT_ID IN (SELECT ENROLLMENT_ID FROM aadhaar WHERE OCCUPATION ILIKE '%farmer%')`. Only use FATHER_NAME_EN when explicitly asked for the name of the father (e.g., "sons of Ramesh").
- NULL HANDLING RULE: For "no bank account" / "unbanked" queries, use: `WHERE (BANK IS NULL OR BANK = '' OR ACCOUNT_NO IS NULL OR ACCOUNT_NO = '')`. Never use a JOIN to find missing records — use IS NULL.
- QUANTIFIER → SQL OPERATOR MAPPING (follow this exactly):
  * "at least N" / "minimum N" / "no fewer than N" → HAVING COUNT(*) >= N
  * "more than N" / "greater than N" / "above N" / "over N" → HAVING COUNT(*) > N
  * "exactly N" / "just N" / "precisely N" → HAVING COUNT(*) = N
  * "at most N" / "no more than N" / "maximum N" / "fewer than N" / "less than N" → HAVING COUNT(*) <= N
  * "fewer than N" / "under N" / "below N" → HAVING COUNT(*) < N
"""

# ── Few-shot examples ─────────────────────────────────────────────────────────
_FEW_SHOTS: list[tuple[str, str | None, str]] = [
    (
        "How many total members are there by district?",
        None,
        f'{{"query": "SELECT DISTRICT_NAME_ENG, SUM(total_members) AS total\\nFROM read_parquet(\'data/district_rollup.parquet\')\\nGROUP BY DISTRICT_NAME_ENG\\nORDER BY total DESC"}}',
    ),
    (
        "List districts and count members in each district, ordered by most members.",
        None,
        f'{{"query": "SELECT DISTRICT_NAME_ENG, SUM(total_members) AS member_count\\nFROM read_parquet(\'data/district_rollup.parquet\')\\nGROUP BY DISTRICT_NAME_ENG\\nORDER BY member_count DESC"}}',
    ),
    (
        "Show female members above age 60 from rural areas.",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, DISTRICT_NAME_ENG, VILL_NAME_ENG\\nFROM {TABLE_NAME}\\nWHERE GENDER = \'Female\'\\n  AND AGE > 60\\n  AND IS_RURAL = \'1\'\\nLIMIT 500"}}',
    ),
    (
        "What is the average income by caste category?",
        None,
        f'{{"query": "SELECT CASTE_CATEGORY, ROUND(AVG(INCOME), 2) AS avg_income\\nFROM {TABLE_NAME}\\nWHERE INCOME IS NOT NULL\\nGROUP BY CASTE_CATEGORY\\nORDER BY avg_income DESC"}}',
    ),
    (
        "Show data of yadavs",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE (CASTE ILIKE \'%yadav%\' OR NAME_EN ILIKE \'%yadav%\')\\nLIMIT 500"}}',
    ),
    (
        "Give me data of all rajputs",
        "(CASTE IN ('राजपूत ', 'RAJPUT', 'RAJPOOT') OR NAME_EN ILIKE '%RAJPUT%' OR NAME_EN ILIKE '%RAJPOOT%')",
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE (CASTE IN (\'राजपूत \', \'RAJPUT\', \'RAJPOOT\') OR NAME_EN ILIKE \'%RAJPUT%\' OR NAME_EN ILIKE \'%RAJPOOT%\')\\nLIMIT 500"}}',
    ),
    (
        "How many members have bank account with SBI?",
        None,
        f'{{"query": "SELECT COUNT(*) AS sbi_members\\nFROM {TABLE_NAME}\\nWHERE BANK ILIKE \'%State Bank of India%\'\\n   OR IFSC_CODE ILIKE \'SBIN%\'"}}',
    ),
    (
        "Show enrollment IDs with more than 5 members in a household.",
        None,
        f'{{"query": "SELECT ENROLLMENT_ID, COUNT(*) AS member_count\\nFROM {TABLE_NAME}\\nGROUP BY ENROLLMENT_ID\\nHAVING member_count > 5\\nORDER BY member_count DESC\\nLIMIT 500"}}',
    ),
    (
        "how many households with more than 2 children?",
        None,
        f'{{"query": "SELECT COUNT(*) AS household_count\\nFROM (\\n    SELECT ENROLLMENT_ID\\n    FROM {TABLE_NAME}\\n    WHERE RELATION_WITH_HOF ILIKE \'%Son%\' OR RELATION_WITH_HOF ILIKE \'%Daughter%\'\\n    GROUP BY ENROLLMENT_ID\\n    HAVING COUNT(*) > 2\\n)"}}',
    ),
    (
        "show list of all households with 1 or more children",
        None,
        f'{{"query": "SELECT ENROLLMENT_ID\\nFROM {TABLE_NAME}\\nWHERE RELATION_WITH_HOF ILIKE \'%Son%\' OR RELATION_WITH_HOF ILIKE \'%Daughter%\'\\nGROUP BY ENROLLMENT_ID\\nHAVING COUNT(*) >= 1\\nLIMIT 500"}}',
    ),
    (
        "Find members born between 1990 and 2000 who are graduates.",
        None,
        f'{{"query": "SELECT NAME_EN, DOB, GENDER, DISTRICT_NAME_ENG, EDUCATION\\nFROM {TABLE_NAME}\\nWHERE DOB >= DATE \'1990-01-01\'\\n  AND DOB <= DATE \'2000-12-31\'\\n  AND EDUCATION ILIKE \'%graduate%\'\\nLIMIT 500"}}',
    ),
    (
        "How many members belong to minority communities like Muslim or Jain?",
        None,
        f'{{"query": "SELECT MINORITY, COUNT(*) AS count\\nFROM {TABLE_NAME}\\nWHERE MINORITY IS NOT NULL\\nGROUP BY MINORITY\\nORDER BY count DESC"}}',
    ),
    # ── Quantifier pattern examples ───────────────────────────────────────────
    (
        "How many families have at least 1 son?",
        None,
        f'{{"query": "SELECT COUNT(*) AS family_count\\nFROM (\\n    SELECT ENROLLMENT_ID\\n    FROM {TABLE_NAME}\\n    WHERE RELATION_WITH_HOF ILIKE \'%Son%\'\\n    GROUP BY ENROLLMENT_ID\\n    HAVING COUNT(*) >= 1\\n)"}}',
    ),
    (
        "How many households have exactly 2 daughters?",
        None,
        f'{{"query": "SELECT COUNT(*) AS household_count\\nFROM (\\n    SELECT ENROLLMENT_ID\\n    FROM {TABLE_NAME}\\n    WHERE RELATION_WITH_HOF ILIKE \'%Daughter%\'\\n    GROUP BY ENROLLMENT_ID\\n    HAVING COUNT(*) = 2\\n)"}}',
    ),
    (
        "How many families have at most 3 members?",
        None,
        f'{{"query": "SELECT COUNT(*) AS family_count\\nFROM (\\n    SELECT ENROLLMENT_ID\\n    FROM {TABLE_NAME}\\n    GROUP BY ENROLLMENT_ID\\n    HAVING COUNT(*) <= 3\\n)"}}',
    ),
    (
        "How many households have more than 2 children?",
        None,
        f'{{"query": "SELECT COUNT(*) AS household_count\\nFROM (\\n    SELECT ENROLLMENT_ID\\n    FROM {TABLE_NAME}\\n    WHERE RELATION_WITH_HOF ILIKE \'%Son%\' OR RELATION_WITH_HOF ILIKE \'%Daughter%\'\\n    GROUP BY ENROLLMENT_ID\\n    HAVING COUNT(*) > 2\\n)"}}',
    ),
    # ── Multi-location (Issue 2) ───────────────────────────────────────────
    (
        "Show people from Srinagar and Beejasar",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE VILL_NAME_ENG ILIKE \'%Srinagar%\'\\n   OR VILL_NAME_ENG ILIKE \'%Beejasar%\'\\nLIMIT 500"}}',
    ),
    (
        "List members from Jaipur or Jodhpur district",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE DISTRICT_NAME_ENG ILIKE \'%Jaipur%\'\\n   OR DISTRICT_NAME_ENG ILIKE \'%Jodhpur%\'\\nLIMIT 500"}}',
    ),
    # ── Family size / membership (Issue 3) ───────────────────────────────
    (
        "Show families with more than 5 members",
        None,
        f'{{"query": "SELECT ENROLLMENT_ID, COUNT(*) AS member_count\\nFROM {TABLE_NAME}\\nGROUP BY ENROLLMENT_ID\\nHAVING COUNT(*) > 5\\nORDER BY member_count DESC\\nLIMIT 500"}}',
    ),
    (
        "Which family has the most members?",
        None,
        f'{{"query": "SELECT ENROLLMENT_ID, COUNT(*) AS member_count\\nFROM {TABLE_NAME}\\nGROUP BY ENROLLMENT_ID\\nORDER BY member_count DESC\\nLIMIT 1"}}',
    ),
    # ── Relational (parent-child) queries (Issue 4) ───────────────────────
    (
        "Show children of members named Rahul",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE FATHER_NAME_EN ILIKE \'%Rahul%\'\\nLIMIT 500"}}',
    ),
    (
        "Who are the daughters of women named Priya?",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE MOTHER_NAME_EN ILIKE \'%Priya%\'\\n  AND RELATION_WITH_HOF ILIKE \'%Daughter%\'\\nLIMIT 500"}}',
    ),
    # ── No bank account (Issue 5) ─────────────────────────────────────
    (
        "Show members who do not have a bank account",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE (BANK IS NULL OR BANK = \'\' OR ACCOUNT_NO IS NULL OR ACCOUNT_NO = \'\')\\nLIMIT 500"}}',
    ),
    # ── Bank abbreviation examples (Issue 1) ────────────────────────────
    (
        "Show members with SBI account",
        "(BANK ILIKE '%State Bank of India%' OR IFSC_CODE ILIKE 'SBIN%')",
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME, BANK, MARITAL_STATUS\\nFROM {TABLE_NAME}\\nWHERE (BANK ILIKE \'%State Bank of India%\' OR IFSC_CODE ILIKE \'SBIN%\')\\nLIMIT 500"}}',
    ),
    (
        "How many members have PNB bank?",
        "(BANK ILIKE '%Punjab National Bank%' OR IFSC_CODE ILIKE 'PUNB%')",
        f'{{"query": "SELECT COUNT(*) AS pnb_members\\nFROM {TABLE_NAME}\\nWHERE (BANK ILIKE \'%Punjab National Bank%\' OR IFSC_CODE ILIKE \'PUNB%\')"}}',
    ),
    # ── Ambiguous location (AMBIGUOUS LOCATION RULE) ──────────────────────
    (
        "show data of people from 2rpm",
        None,
        f'{{"query": "SELECT NAME_EN, AGE, GENDER, CASTE_CATEGORY, DISTRICT_NAME_ENG, VILL_NAME_ENG, OCCUPATION, INCOME\\nFROM {TABLE_NAME}\\nWHERE (VILL_NAME_ENG ILIKE \'%2rpm%\' OR GP_NAME_ENG ILIKE \'%2rpm%\' OR BLOCK_NAME_ENG ILIKE \'%2rpm%\' OR DISTRICT_NAME_ENG ILIKE \'%2rpm%\')\\nLIMIT 500"}}',
    ),
]


# ── Builder ───────────────────────────────────────────────────────────────────

def build_prompt(
    user_question: str,
    schema_cols: list["ColumnDef"],
    domain_hints: list[str] | None = None,
) -> str:
    """
    Assemble the full LLM prompt string with strict static ordering for KV caching:
    System -> Schema -> Few-Shots -> Dynamic Hints -> User Question.
    """
    parts = [_SYSTEM]

    # 1. Pruned Schema (Static-ish per schema size)
    parts.append("### DATASET SCHEMA")
    for col in schema_cols:
        parts.append(f"- {col['name']} ({col['dtype']}): {col['description']}")
    parts.append("")

    # 2. Few-Shot Examples (100% Static)
    parts.append("### EXAMPLES")
    for q, h, sql in _FEW_SHOTS:
        parts.append(f"Question: {q}")
        if h:
            parts.append(f"Domain Hint: Use {h}")
        parts.append(f"SQL:\n{sql}\n")

    # 3. Dynamic Domain Hints (pushed to the bottom)
    if domain_hints:
        parts.append("### DOMAIN HINTS FOR CURRENT QUESTION")
        parts.append("Use these exact SQL conditions where applicable:")
        for hint in domain_hints:
            parts.append(f"- {hint}")
        parts.append("")

    # 4. The User's Question (absolute bottom)
    parts.append(f"### CURRENT QUESTION\nQuestion: {user_question}\nJSON Output:\n")

    return "\n".join(parts)


def build_correction_prompt(
    user_question: str,
    schema_cols: list["ColumnDef"],
    failed_sql: str,
    error_msg: str,
) -> str:
    """
    Build a retry prompt that includes the failed SQL and the error message.

    Args:
        user_question: Original user question.
        schema_cols:   Same pruned schema columns.
        failed_sql:    The SQL that failed validation or execution.
        error_msg:     The error/reason from sqlglot or DuckDB.

    Returns:
        Correction prompt string.
    """
    base = build_prompt(user_question, schema_cols)
    correction = (
        f"\n-- The previous attempt failed:\n"
        f"-- SQL: {failed_sql}\n"
        f"-- Error: {error_msg}\n"
        f"-- Please fix the SQL and output ONLY the corrected query inside the JSON object.\n"
        f"### Corrected JSON Output\n"
    )
    return base + correction


def build_scaffold_prompt(
    user_question: str,
    schema_cols: list["ColumnDef"],
    scaffold: "PartialScaffold",
) -> str:
    """
    Build a shorter, scaffold-assisted LLM prompt.

    The deterministic layer has already pre-built the known WHERE conditions.
    This prompt:
    1. Uses a compressed system instruction (no need to re-explain already-handled rules).
    2. Only includes schema columns the LLM still needs to reason about.
    3. Injects the pre-built WHERE clause as a HARD CONSTRAINT the LLM MUST keep verbatim.
    4. Focuses the LLM only on the unmapped tokens.

    This typically reduces prompt token count by 40-60% vs. the full prompt.
    """
    from llm.fast_sql import PartialScaffold  # local import to avoid circular

    # ── Compressed system instruction ────────────────────────────────────────
    system = f"""\
You are a DuckDB SQL generator for a government Aadhaar welfare dataset.
Table name: `{TABLE_NAME}`

STRICT RULES:
- Output ONLY this exact JSON format: {{"query": "SELECT ..."}}
- No markdown, no code fences, no explanation.
- DuckDB SQL only (ILIKE, TRY_CAST, REGEXP_MATCHES).
- Only reference columns from the schema below. NEVER hallucinate column names.
- String comparisons: use ILIKE or LOWER().
- Add LIMIT 500 unless this is a pure aggregation (COUNT, SUM, AVG, GROUP BY without LIMIT).
- DO NOT include semicolons anywhere in the SQL.
- RELATIONAL QUERY RULE: To find members based on traits of their family/parents (e.g., "sons of farmers", "wives of men in Jaipur"), you MUST use an IN subquery on ENROLLMENT_ID. DO NOT USE JOINs or SELF-JOINs. ALWAYS select from a single 'aadhaar' table. Example: `WHERE RELATION_WITH_HOF ILIKE '%Son%' AND ENROLLMENT_ID IN (SELECT ENROLLMENT_ID FROM aadhaar WHERE OCCUPATION ILIKE '%farmer%')`. Only use FATHER_NAME_EN when explicitly asked for the name of the father (e.g., "sons of Ramesh").
- For date arithmetic use CURRENT_DATE - INTERVAL N YEAR. NEVER use DATE_SUB(). Do not use AGE as a function (AGE is a numeric column in the schema).
"""

    parts = [system]

    # ── Pruned schema — only columns NOT already covered ─────────────────────
    uncovered_cols = [
        col for col in schema_cols
        if col["name"] not in scaffold.covered_columns
    ]
    # Always include a small set of core display columns even if covered
    core_display = {"NAME_EN", "AGE", "GENDER", "DISTRICT_NAME_ENG", "VILL_NAME_ENG"}
    extra_cols = [
        col for col in schema_cols
        if col["name"] in core_display and col["name"] not in {c["name"] for c in uncovered_cols}
    ]
    final_cols = uncovered_cols + extra_cols

    if final_cols:
        parts.append("### RELEVANT SCHEMA COLUMNS")
        for col in final_cols:
            parts.append(f"- {col['name']} ({col['dtype']}): {col['description']}")
        parts.append("")

    # ── Hard-wired WHERE constraint (the LLM MUST keep this verbatim) ────────
    if scaffold.known_conditions:
        parts.append("### PRE-BUILT WHERE CONDITIONS (MANDATORY — include these EXACTLY as-is)")
        parts.append("The following SQL conditions have already been determined. You MUST include all of them in your WHERE clause verbatim:")
        for cond in scaffold.known_conditions:
            parts.append(f"  {cond}")
        parts.append("")
        parts.append(f"Your WHERE clause MUST start with:\n  {scaffold.where_clause}")
        parts.append("")

    # ── Hint about what the LLM still needs to resolve ────────────────────────
    if scaffold.unmapped_tokens:
        parts.append("### WHAT YOU STILL NEED TO RESOLVE")
        parts.append(f"The following part(s) of the user's query were not pre-mapped and you must handle them:")
        parts.append(f"  {' '.join(scaffold.unmapped_tokens)}")
        parts.append("")

    # ── Intent hint ──────────────────────────────────────────────────────────
    if scaffold.is_count:
        parts.append("### INTENT: The user wants a COUNT (aggregate), not a row listing.")
    elif scaffold.groupby_col:
        parts.append(f"### INTENT: The user wants results broken down by {scaffold.groupby_col} (GROUP BY).")
    elif scaffold.is_list:
        parts.append("### INTENT: The user wants a list of individual records (SELECT rows, LIMIT 500).")
    parts.append("")

    # ── The question ─────────────────────────────────────────────────────────
    parts.append(f"### QUESTION\nQuestion: {user_question}\nJSON Output:\n")

    return "\n".join(parts)

