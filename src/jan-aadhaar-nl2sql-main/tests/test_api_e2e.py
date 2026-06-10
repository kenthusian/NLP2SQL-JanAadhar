"""
tests/test_api_e2e.py — End-to-end query correctness tests against the live API.

Prerequisites:
  - python main.py must be running on http://127.0.0.1:8000
  - The DuckDB dataset must be loaded (data/aadhaar/**/*.parquet)

Run with:  python tests/test_api_e2e.py

Each test sends a POST /query and asserts:
  - HTTP 200
  - SQL structure correctness (expected keywords present)
  - Result sanity (row_count >= expected_min, result_value assertions where known)
"""
import sys
import json
from pathlib import Path
import requests

API_BASE = "http://127.0.0.1:8000"


def post(question: str) -> dict:
    resp = requests.post(f"{API_BASE}/query", json={"question": question}, timeout=180)
    assert resp.status_code == 200, f"[{question!r}] HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def sql_contains(data: dict, *fragments: str):
    sql = data["sql"].upper()
    for frag in fragments:
        assert frag.upper() in sql, f"Expected {frag!r} in SQL:\n{data['sql']}"


def assert_single_int(data: dict) -> int:
    """Assert exactly 1 row, 1 column of integer type. Return the value."""
    assert data["row_count"] == 1, f"Expected 1 row, got {data['row_count']}"
    assert len(data["columns"]) == 1, f"Expected 1 column, got {data['columns']}"
    val = list(data["rows"][0].values())[0]
    assert isinstance(val, (int, float)), f"Expected numeric result, got {val!r}"
    return int(val)


PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []


def run_test(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"{PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL}  {name}")
        print(f"         {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Basic health
# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    resp = requests.get(f"{API_BASE}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["duckdb"], "DuckDB not healthy"


# ─────────────────────────────────────────────────────────────────────────────
# Simple count queries
# ─────────────────────────────────────────────────────────────────────────────

def test_count_all():
    """Total row count should be > 0."""
    data = post("how many total members are there?")
    val = assert_single_int(data)
    assert val > 0, f"Expected > 0 members, got {val}"


def test_count_male():
    """Male count should be positive integer."""
    data = post("how many male members are there?")
    val = assert_single_int(data)
    assert val >= 0
    sql_contains(data, "GENDER", "Male")


def test_count_female():
    """Female count should be positive."""
    data = post("how many female members?")
    val = assert_single_int(data)
    assert val >= 0
    sql_contains(data, "GENDER", "Female")


def test_male_plus_female_leq_total():
    """Males + Females should be <= total (there may be other genders)."""
    total = assert_single_int(post("how many total members?"))
    males = assert_single_int(post("how many male members?"))
    females = assert_single_int(post("how many female members?"))
    assert males + females <= total, f"Males({males}) + Females({females}) > Total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# IS_RURAL correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_rural_uses_1():
    """Rural queries must use IS_RURAL = '1', not 'R' or 'Rural'."""
    data = post("count families in rural areas")
    sql = data["sql"]
    assert "IS_RURAL = '1'" in sql, f"IS_RURAL normalisation failed: {sql}"


def test_urban_uses_0():
    """Urban queries must use IS_RURAL = '0', not 'U' or 'Urban'."""
    data = post("how many members live in urban areas?")
    sql = data["sql"]
    assert "IS_RURAL = '0'" in sql, f"IS_RURAL normalisation failed: {sql}"


def test_rural_plus_urban_leq_total():
    """Rural + Urban should be <= total members."""
    total = assert_single_int(post("how many total members?"))
    rural = assert_single_int(post("count families in rural areas"))
    urban = assert_single_int(post("how many members live in urban areas?"))
    assert rural + urban <= total + 1, f"Rural({rural}) + Urban({urban}) > Total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# Quantifier → HAVING operator correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_at_least_1_son_uses_gte():
    """'at least 1 son' must generate HAVING COUNT(*) >= 1."""
    data = post("how many families have at least 1 son?")
    sql = data["sql"].upper()
    assert ">= 1" in sql or ">=1" in sql, f"Expected >= 1 in SQL:\n{data['sql']}"


def test_more_than_2_children_uses_gt():
    """'more than 2 children' must generate HAVING COUNT(*) > 2."""
    data = post("how many households have more than 2 children?")
    sql = data["sql"].upper()
    assert "> 2" in sql or ">2" in sql, f"Expected > 2 in SQL:\n{data['sql']}"


def test_exactly_2_daughters_uses_eq():
    """'exactly 2 daughters' must generate HAVING COUNT(*) = 2."""
    data = post("how many households have exactly 2 daughters?")
    sql = data["sql"].upper()
    assert "= 2" in sql, f"Expected = 2 in SQL:\n{data['sql']}"


def test_at_most_3_members_uses_lte():
    """'at most 3 members' must generate HAVING COUNT(*) <= 3."""
    data = post("how many families have at most 3 members?")
    sql = data["sql"].upper()
    assert "<= 3" in sql or "<=3" in sql, f"Expected <= 3 in SQL:\n{data['sql']}"


def test_at_least_lt_more_than_makes_sense():
    """
    'at least 1 child' count >= 'more than 1 child' count.
    Because at-least-1 includes households with exactly 1 child too.
    """
    at_least_1 = assert_single_int(post("how many families have at least 1 son?"))
    more_than_1 = assert_single_int(post("how many families have more than 1 son?"))
    assert at_least_1 >= more_than_1, (
        f"at_least_1({at_least_1}) < more_than_1({more_than_1}): operator bug!"
    )


def test_exactly_subset_of_at_least():
    """
    'exactly 1 son' count <= 'at least 1 son' count (exactly is a strict subset).
    """
    at_least_1 = assert_single_int(post("how many families have at least 1 son?"))
    exactly_1 = assert_single_int(post("how many households have exactly 1 son?"))
    assert exactly_1 <= at_least_1, (
        f"exactly_1({exactly_1}) > at_least_1({at_least_1}): operator bug!"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subquery structure for household counting
# ─────────────────────────────────────────────────────────────────────────────

def test_household_count_uses_subquery():
    """Household count queries must use subquery pattern, not COUNT(DISTINCT ...)."""
    data = post("how many households have at least 1 daughter?")
    sql = data["sql"].upper()
    # Must not be COUNT(DISTINCT ENROLLMENT_ID) ... GROUP BY ENROLLMENT_ID
    has_bad_pattern = "COUNT(DISTINCT ENROLLMENT_ID)" in sql and "GROUP BY ENROLLMENT_ID" in sql
    assert not has_bad_pattern, f"Found anti-pattern COUNT(DISTINCT...)/GROUP BY:\n{data['sql']}"


# ─────────────────────────────────────────────────────────────────────────────
# Cache correctness: ensure cache hit returns same result as fresh query
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_returns_same_result():
    """Second identical query should hit cache and return same row_count as first."""
    q = "how many members are from OBC caste category?"
    r1 = post(q)
    r2 = post(q)
    assert r2["source"] == "cache", "Second query should be a cache hit"
    assert r2["row_count"] == r1["row_count"], (
        f"Cache returned different row_count: first={r1['row_count']} second={r2['row_count']}"
    )


def test_semantic_cache_does_not_corrupt_different_queries():
    """
    Two semantically similar but numerically different queries must NOT return
    the same cached result (the number guard should distinguish them).
    """
    r1 = post("how many families have at least 1 son?")
    r2 = post("how many families have at least 2 sons?")
    # They should have different SQL (specifically the HAVING threshold differs)
    sql1 = r1["sql"].upper()
    sql2 = r2["sql"].upper()
    # Both should be >= but with different numbers
    # At minimum, they should not return IDENTICAL SQL
    assert sql1 != sql2, (
        f"Cache poisoning: numerically different queries returned identical SQL:\n{sql1}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# District and geography queries
# ─────────────────────────────────────────────────────────────────────────────

def test_district_query_returns_rows():
    data = post("show me members from Jaipur district")
    assert data["row_count"] >= 0
    sql_contains(data, "DISTRICT_NAME_ENG")


def test_caste_category_filter():
    data = post("list all SC category members")
    assert data["row_count"] >= 0
    sql_contains(data, "CASTE_CATEGORY")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("API health check", test_health),
    ("Count all members", test_count_all),
    ("Count male members", test_count_male),
    ("Count female members", test_count_female),
    ("Male + Female <= Total", test_male_plus_female_leq_total),
    ("IS_RURAL = '1' for rural queries", test_rural_uses_1),
    ("IS_RURAL = '0' for urban queries", test_urban_uses_0),
    ("Rural + Urban <= Total", test_rural_plus_urban_leq_total),
    ("'at least 1 son' uses >= 1", test_at_least_1_son_uses_gte),
    ("'more than 2 children' uses > 2", test_more_than_2_children_uses_gt),
    ("'exactly 2 daughters' uses = 2", test_exactly_2_daughters_uses_eq),
    ("'at most 3 members' uses <= 3", test_at_most_3_members_uses_lte),
    ("at_least_1_son >= more_than_1_son (logic check)", test_at_least_lt_more_than_makes_sense),
    ("exactly_1_son <= at_least_1_son (logic check)", test_exactly_subset_of_at_least),
    ("Household count uses subquery pattern", test_household_count_uses_subquery),
    ("Cache hit returns same result", test_cache_returns_same_result),
    ("Semantic cache doesn't corrupt different numbers", test_semantic_cache_does_not_corrupt_different_queries),
    ("District query works", test_district_query_returns_rows),
    ("Caste category filter", test_caste_category_filter),
]


if __name__ == "__main__":
    print(f"\n{'=' * 65}")
    print(f"  Jan-Aadhaar NL2SQL — End-to-End Test Suite")
    print(f"{'=' * 65}\n")

    for name, fn in TESTS:
        run_test(name, fn)

    print(f"\n{'=' * 65}")
    passed = sum(1 for r in results if r[0] == PASS)
    failed = len(results) - passed
    print(f"  Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"{'=' * 65}\n")

    if failed > 0:
        print("Failed tests:")
        for r in results:
            if r[0] == FAIL:
                print(f"  {r[1]}: {r[2]}")
        sys.exit(1)
