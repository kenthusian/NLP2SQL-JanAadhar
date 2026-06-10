from __future__ import annotations

import re

from database.schema_metadata import COLUMNS, RAJASTHAN_DISTRICTS_41, RAJASTHAN_CITIES, RAJASTHAN_BLOCKS
from retrieval.schema_retriever import RetrievalResult


_DISTRICTS_LOWER: dict[str, str] = {d.lower(): d for d in RAJASTHAN_DISTRICTS_41}
_CITIES_LOWER: dict[str, str] = {c.lower(): c for c in RAJASTHAN_CITIES}
_BLOCKS_LOWER: dict[str, str] = {b.lower(): b for b in RAJASTHAN_BLOCKS}

# ── Few-shot examples (teach the LLM the schema once) ────────────────────────
_FEW_SHOT = """
Examples:
Q: Show all males in Jaipur
SQL: SELECT member_name, age, gender, district FROM citizen WHERE gender = 'Male' AND district = 'Jaipur';

Q: Count of SC widows above 50
SQL: SELECT COUNT(*) AS count FROM citizen WHERE caste_category = 'SC' AND marital_status = 'Widow' AND age > 50;

Q: Farmers in rural Bikaner
SQL: SELECT member_name, age, occupation, village FROM citizen WHERE occupation LIKE '%Farmer%' AND district = 'Bikaner' AND is_rural = 1;

Q: Members without a bank account
SQL: SELECT member_name, district, age FROM citizen WHERE bank_account IS NULL;

Q: show details of studants in jodhpur
SQL: SELECT member_name, age, gender, district FROM citizen WHERE occupation LIKE '%Student%' AND district = 'Jodhpur';

Q: Average income of OBC citizens in Jodhpur
SQL: SELECT AVG(income) AS avg_income FROM citizen WHERE caste_category = 'OBC' AND district = 'Jodhpur';

Q: How many children (below 18) in each district
SQL: SELECT district, COUNT(*) AS child_count FROM citizen WHERE age < 18 GROUP BY district ORDER BY child_count DESC;

Q: Members of families that have at least one doctor
SQL: SELECT member_name, district, age FROM citizen WHERE jan_aadhaar_family_id IN (SELECT jan_aadhaar_family_id FROM citizen WHERE occupation LIKE '%Doctor%');

Q: show data of all jaat whose sons have account in SBI
SQL: SELECT member_name, age, gender, district FROM citizen WHERE caste LIKE '%Jat%' AND jan_aadhaar_family_id IN (SELECT jan_aadhaar_family_id FROM citizen WHERE relation_with_hof = 'Son' AND bank_name LIKE '%SBI%');
"""


class PromptBuilder:
    def build(
        self,
        result: RetrievalResult,
        previous_error: str | None = None,
        dialect: str = "sqlite",
    ) -> str:

        column_lookup = {col.qualified_name: col for col in COLUMNS}
        column_lines: list[str] = []
        for qn in result.columns:
            col = column_lookup.get(qn)
            if not col:
                continue
            samples = f"; e.g. {', '.join(col.sample_values)}" if col.sample_values else ""
            column_lines.append(
                f"  - {col.column} ({col.data_type}): {col.description}{samples}"
            )

        # ── District pre-classification ───────────────────────────────────────
        location_hints = _extract_location_hints(result.question)
        location_notes: list[str] = []
        for token in location_hints:
            token_lower = token.lower().strip()
            canonical_d = _DISTRICTS_LOWER.get(token_lower)
            canonical_c = _CITIES_LOWER.get(token_lower)
            canonical_b = _BLOCKS_LOWER.get(token_lower)
            
            if canonical_d:
                location_notes.append(f"  - '{token}' is a known district → use: district = '{canonical_d}'")
            elif canonical_c:
                location_notes.append(f"  - '{token}' is a known city → use: city = '{canonical_c}'")
            elif canonical_b:
                location_notes.append(f"  - '{token}' is a known block → use: block = '{canonical_b}'")
            else:
                location_notes.append(f"  - '{token}' is an unknown location → use: district = '{token}' (the system will automatically search all location columns)")
        location_block = ""
        if location_notes:
            location_block = "\nLocation pre-classification (use exactly as written):\n" + "\n".join(location_notes)

        error_block = (
            f"\nPrevious SQL was invalid: {previous_error}\nFix it.\n"
            if previous_error else ""
        )

        col_section = "\n".join(column_lines) if column_lines else "  (all columns of citizen)"

        return f"""You are a SQLite SQL generator for a single flat table called `citizen`.
Return ONLY the SQL. No markdown, no explanation, no comments.
Generate exactly one read-only SELECT statement.
Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any DDL/DML.
You may use subqueries on `jan_aadhaar_family_id` for queries involving multiple family members. Do NOT use JOIN.
Use only the columns listed below. Do not invent columns.
{_FEW_SHOT}
Table: citizen
Columns:
{col_section}
{location_block}

Rules:
- gender: stored as 'Male' or 'Female'. boy/boys/man/men → 'Male'; girl/girls/woman/women → 'Female'.
- widow/widowed → marital_status = 'Widow'; unmarried/single → 'Unmarried'.
- senior citizen / elderly / old age → age >= 60; child/minor → age < 18; adult → age >= 18.
- above N / older than N → age > N; below N / younger than N → age < N.
- caste_category: SC, ST, OBC, GEN only. 'general'/'open' → 'GEN'.
- is_rural: 1 = rural, 0 = urban.
- education: 'illiterate' stored lowercase — use LOWER(education) = 'illiterate' or LIKE '%illiterate%'.
- bank account: 'no bank account'/'unbanked' → bank_account IS NULL.
- For name searches always use LIKE: member_name LIKE '%Raj%'.
- caste searches: always use LIKE: caste LIKE '%Jat%'.
- bank_name: always UPPER(bank_name) LIKE '%SBI%'.
- how many / count of → COUNT(*); total income → SUM(income); average → AVG(income).
- If question asks to 'show' or 'list', SELECT identifying columns (member_name, district, age etc.), not COUNT.
- FAMILY RELATIONS: If a query asks about "families where...", or "whose sons have...", or properties of multiple members simultaneously, you MUST use a subquery: `jan_aadhaar_family_id IN (SELECT jan_aadhaar_family_id FROM citizen WHERE ...)`
- TYPO CORRECTION: If a user word is misspelled (e.g. 'studant', 'femail', 'farms'), aggressively autocorrect it to match the closest known schema values listed above or use a very permissive LIKE clause.
{error_block}
Question: {result.question}

SQL:"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_location_hints(question: str) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for prep in ("in", "from", "at"):
        for m in re.finditer(
            rf"\b{prep}\s+([A-Za-z][A-Za-z\s\-]{{1,30}}?)(?:\s+(?:and|or|where|who|that|which|with|having|are|is)\b|[,.]|$)",
            question,
            re.IGNORECASE,
        ):
            raw = m.group(1).strip()
            token = " ".join(raw.split()[:3])
            key = token.lower()
            stopwords = {"age", "years", "male", "female", "boys", "girls"}
            if key in stopwords or key in seen or not token:
                continue
            seen.add(key)
            hints.append(token)
    return hints
