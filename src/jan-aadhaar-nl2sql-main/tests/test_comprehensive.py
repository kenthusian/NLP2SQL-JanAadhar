"""
tests/test_comprehensive.py — Exhaustive NL2SQL query test suite.

Tests every category of query the system is expected to handle:
  1.  Simple counts (total, by gender, by area)
  2.  Filter queries (gender, age, caste, education, occupation, income, bank)
  3.  Household/family counting with all quantifiers
  4.  Aggregation queries (SUM, AVG, GROUP BY, ORDER BY)
  5.  Date/age range queries
  6.  Geography (district, block, village, GP)
  7.  Caste & community
  8.  Education & occupation
  9.  Income & BPL
  10. Banking (bank name, IFSC, account)
  11. Minority queries
  12. Marital status
  13. Combined / cross-dimensional queries
  14. IS_RURAL normalisation (correctness guard)
  15. Quantifier operator correctness (logic guard)
  16. Cache integrity (hit returns same result)
  17. Cache isolation (different numbers → different SQL)
  18. Typo tolerance (misspelled inputs)

Run with:
    python tests/test_comprehensive.py
    -- or --
    python tests/test_comprehensive.py --stop-on-first-fail

Usage tips:
  - Set SKIP_SLOW=1 env var to skip multi-query logical consistency checks
  - Set TIMEOUT=<seconds> to change per-query timeout (default: 180)
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path

import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("TIMEOUT", "180"))
SKIP_SLOW = os.getenv("SKIP_SLOW", "0") == "1"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
WARN = "[WARN]"

results: list[tuple] = []
stop_on_first_fail = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def post(question: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/query",
        json={"question": question},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise AssertionError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def get_int(data: dict) -> int:
    assert data["row_count"] >= 1, f"Expected >=1 row, got 0. SQL: {data['sql']}"
    val = list(data["rows"][0].values())[0]
    return int(val)


def sql_has(data: dict, *fragments: str) -> bool:
    sql = data["sql"].upper()
    return all(f.upper() in sql for f in fragments)


def assert_sql_has(data: dict, *fragments: str):
    for f in fragments:
        assert f.upper() in data["sql"].upper(), \
            f"Expected {f!r} in SQL:\n{data['sql']}"


def run_test(name: str, fn, skip: bool = False):
    global stop_on_first_fail
    if skip:
        results.append((SKIP, name, ""))
        print(f"{SKIP}  {name}")
        return
    try:
        t0 = time.perf_counter()
        fn()
        ms = round((time.perf_counter() - t0) * 1000)
        results.append((PASS, name, ""))
        print(f"{PASS}  {name}  ({ms}ms)")
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL}  {name}")
        print(f"        → {e}")
        if stop_on_first_fail:
            sys.exit(1)
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))
        print(f"{FAIL}  {name}")
        print(f"        → {type(e).__name__}: {e}")
        if stop_on_first_fail:
            sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Health
# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{API_BASE}/health", timeout=10).json()
    assert r["duckdb"], "DuckDB not healthy"
    assert r["status"] in ("ok", "degraded")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Simple counts
# ─────────────────────────────────────────────────────────────────────────────

def test_count_total():
    d = post("how many total members are there?")
    v = get_int(d)
    assert v > 0, f"Total members = {v}"


def test_count_male():
    d = post("how many male members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "GENDER")


def test_count_female():
    d = post("how many female members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "GENDER")


def test_male_plus_female_le_total():
    if SKIP_SLOW: return
    total = get_int(post("how many total members are there?"))
    males = get_int(post("how many male members?"))
    females = get_int(post("how many female members?"))
    assert males + females <= total + 5, \
        f"Males({males}) + Females({females}) > Total({total})"


def test_count_families():
    d = post("how many families are there?")
    assert d["row_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. IS_RURAL correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_rural_is_1():
    d = post("how many members live in rural areas?")
    assert "IS_RURAL = '1'" in d["sql"], f"Bad IS_RURAL in: {d['sql']}"


def test_urban_is_0():
    d = post("how many members live in urban areas?")
    assert "IS_RURAL = '0'" in d["sql"], f"Bad IS_RURAL in: {d['sql']}"


def test_village_maps_to_rural():
    d = post("count members from villages")
    assert "IS_RURAL = '1'" in d["sql"], f"Village should map to IS_RURAL='1': {d['sql']}"


def test_city_maps_to_urban():
    d = post("show members from city areas")
    assert "IS_RURAL = '0'" in d["sql"], f"City should map to IS_RURAL='0': {d['sql']}"


def test_rural_plus_urban_le_total():
    if SKIP_SLOW: return
    total = get_int(post("how many total members are there?"))
    rural = get_int(post("how many members in rural areas?"))
    urban = get_int(post("how many members in urban areas?"))
    assert rural + urban <= total + 10, \
        f"Rural({rural}) + Urban({urban}) > Total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantifier → HAVING operator correctness
# ─────────────────────────────────────────────────────────────────────────────

def _check_having_op(question: str, expected_op: str, expected_num: str):
    d = post(question)
    sql = d["sql"].upper()
    pattern = f"HAVING COUNT(*) {expected_op} {expected_num}"
    assert pattern in sql, \
        f"Expected '{pattern}' in SQL for question {question!r}:\n{d['sql']}"


def test_quantifier_at_least_1():
    _check_having_op("how many families have at least 1 son?", ">=", "1")


def test_quantifier_at_least_2():
    _check_having_op("how many households have at least 2 daughters?", ">=", "2")


def test_quantifier_minimum_3():
    _check_having_op("families with minimum 3 members", ">=", "3")


def test_quantifier_more_than_1():
    _check_having_op("how many families have more than 1 child?", ">", "1")


def test_quantifier_more_than_2():
    _check_having_op("how many households have more than 2 members?", ">", "2")


def test_quantifier_greater_than_3():
    _check_having_op("families with greater than 3 sons", ">", "3")


def test_quantifier_exactly_1():
    _check_having_op("how many households have exactly 1 son?", "=", "1")


def test_quantifier_exactly_2():
    _check_having_op("families with exactly 2 daughters", "=", "2")


def test_quantifier_at_most_3():
    _check_having_op("how many households have at most 3 members?", "<=", "3")


def test_quantifier_no_more_than_2():
    _check_having_op("families with no more than 2 children", "<=", "2")


def test_quantifier_fewer_than_3():
    _check_having_op("households with fewer than 3 members", "<", "3")


def test_quantifier_less_than_2():
    _check_having_op("families with less than 2 sons", "<", "2")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quantifier logical consistency checks
# ─────────────────────────────────────────────────────────────────────────────

def test_at_least_1_ge_more_than_1():
    """at-least-1 count must be >= more-than-1 count"""
    if SKIP_SLOW: return
    a = get_int(post("how many families have at least 1 son?"))
    b = get_int(post("how many families have more than 1 son?"))
    assert a >= b, f"at_least_1({a}) < more_than_1({b}) — operator bug!"


def test_exactly_1_subset_of_at_least_1():
    """exactly-1 count must be <= at-least-1 count"""
    if SKIP_SLOW: return
    exact = get_int(post("how many households have exactly 1 son?"))
    atleast = get_int(post("how many families have at least 1 son?"))
    assert exact <= atleast, f"exactly_1({exact}) > at_least_1({atleast})"


def test_at_most_le_total():
    """at-most-N families cannot exceed total families"""
    if SKIP_SLOW: return
    total = get_int(post("how many total members are there?"))
    atmost5 = get_int(post("how many families have at most 5 members?"))
    assert atmost5 <= total + 10, f"at_most_5({atmost5}) > total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Household counting — subquery pattern required
# ─────────────────────────────────────────────────────────────────────────────

def test_household_count_no_count_distinct_groupby():
    """Must use subquery, not COUNT(DISTINCT ...) ... GROUP BY"""
    d = post("how many households have at least 1 daughter?")
    sql = d["sql"].upper()
    bad = "COUNT(DISTINCT ENROLLMENT_ID)" in sql and "GROUP BY ENROLLMENT_ID" in sql
    assert not bad, f"Found anti-pattern:\n{d['sql']}"


def test_household_son():
    d = post("how many families have at least 1 son?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "ENROLLMENT_ID", "HAVING")


def test_household_daughter():
    d = post("how many families have at least 1 daughter?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "ENROLLMENT_ID", "HAVING")


def test_household_children_exact():
    d = post("how many families have exactly 2 children?")
    assert d["row_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Caste & community
# ─────────────────────────────────────────────────────────────────────────────

def test_caste_category_obc():
    d = post("how many OBC members are there?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "CASTE_CATEGORY")


def test_caste_category_sc():
    d = post("list members from scheduled caste")
    assert d["row_count"] >= 0
    assert_sql_has(d, "CASTE_CATEGORY")


def test_caste_category_st():
    d = post("how many scheduled tribe members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "CASTE_CATEGORY")


def test_caste_category_gen():
    d = post("count general category members")
    assert d["row_count"] >= 1
    assert_sql_has(d, "CASTE_CATEGORY")


def test_caste_specific_rajput():
    d = post("show data of rajputs")
    assert d["row_count"] >= 0
    assert_sql_has(d, "CASTE")


def test_caste_specific_yadav():
    d = post("list all yadavs")
    assert d["row_count"] >= 0
    assert_sql_has(d, "CASTE")


def test_caste_categories_sum_le_total():
    """SC + ST + OBC + GEN <= total (some may be NULL)"""
    if SKIP_SLOW: return
    total = get_int(post("how many total members are there?"))
    sc = get_int(post("how many SC members?"))
    st = get_int(post("how many ST members?"))
    obc = get_int(post("how many OBC members?"))
    gen = get_int(post("how many general category members?"))
    assert sc + st + obc + gen <= total + 10, \
        f"Caste sum ({sc+st+obc+gen}) > total ({total})"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Age & demographic
# ─────────────────────────────────────────────────────────────────────────────

def test_senior_citizens():
    d = post("how many senior citizens are there?")
    assert d["row_count"] >= 1
    sql = d["sql"].upper()
    assert "AGE" in sql and ("60" in sql or ">=" in sql)


def test_minors():
    d = post("how many minors under 18?")
    assert d["row_count"] >= 1
    sql = d["sql"].upper()
    assert "AGE" in sql


def test_adults():
    d = post("show all adult members above 18")
    assert d["row_count"] >= 1
    assert_sql_has(d, "AGE")


def test_age_range():
    d = post("members between age 20 and 30")
    assert d["row_count"] >= 0
    assert_sql_has(d, "AGE")


def test_minors_le_total():
    """Minors must be <= total members"""
    if SKIP_SLOW: return
    total = get_int(post("how many total members?"))
    minors = get_int(post("how many minors?"))
    assert minors <= total, f"minors({minors}) > total({total})"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Education
# ─────────────────────────────────────────────────────────────────────────────

def test_graduates():
    d = post("how many graduates are there?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "EDUCATION")


def test_illiterates():
    d = post("count illiterate members")
    assert d["row_count"] >= 1
    assert_sql_has(d, "EDUCATION")


def test_education_by_district():
    d = post("show graduates by district")
    assert d["row_count"] >= 0
    assert_sql_has(d, "EDUCATION", "DISTRICT_NAME_ENG")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Occupation
# ─────────────────────────────────────────────────────────────────────────────

def test_farmers():
    d = post("how many farmers?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "OCCUPATION")


def test_students():
    d = post("count students")
    assert d["row_count"] >= 1
    assert_sql_has(d, "OCCUPATION")


def test_unemployed():
    d = post("how many unemployed members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "OCCUPATION")


def test_occupation_distribution():
    d = post("show occupation distribution")
    assert d["row_count"] >= 1
    assert_sql_has(d, "OCCUPATION")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Income
# ─────────────────────────────────────────────────────────────────────────────

def test_average_income():
    d = post("what is the average income?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "INCOME", "AVG")


def test_avg_income_by_caste():
    d = post("average income by caste category")
    assert d["row_count"] >= 1
    assert_sql_has(d, "INCOME", "CASTE_CATEGORY")


def test_low_income():
    d = post("show poor families with income below 50000")
    assert d["row_count"] >= 0
    assert_sql_has(d, "INCOME")


def test_high_income():
    d = post("members with income above 100000")
    assert d["row_count"] >= 0
    assert_sql_has(d, "INCOME")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Banking
# ─────────────────────────────────────────────────────────────────────────────

def test_sbi_members():
    d = post("how many members have SBI bank account?")
    assert d["row_count"] >= 1
    sql = d["sql"].upper()
    assert "BANK" in sql or "IFSC" in sql


def test_members_with_no_bank():
    d = post("members without a bank account")
    assert d["row_count"] >= 0
    assert_sql_has(d, "BANK")


def test_bank_distribution():
    d = post("show bank distribution")
    assert d["row_count"] >= 1
    assert_sql_has(d, "BANK")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Marital status
# ─────────────────────────────────────────────────────────────────────────────

def test_married():
    d = post("how many married members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "MARITAL_STATUS")


def test_unmarried():
    d = post("count unmarried members")
    assert d["row_count"] >= 1
    assert_sql_has(d, "MARITAL_STATUS")


def test_widowed():
    d = post("how many widowed members?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "MARITAL_STATUS")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Minority
# ─────────────────────────────────────────────────────────────────────────────

def test_minority_members():
    d = post("how many members belong to minority communities?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "MINORITY")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Geography
# ─────────────────────────────────────────────────────────────────────────────

def test_district_filter():
    d = post("show members from Jaipur district")
    assert d["row_count"] >= 0
    assert_sql_has(d, "DISTRICT_NAME_ENG")


def test_district_count():
    d = post("how many members in each district?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "DISTRICT_NAME_ENG")


def test_block_filter():
    d = post("show members from rural blocks")
    assert d["row_count"] >= 0


def test_village_filter():
    d = post("list members from villages")
    assert d["row_count"] >= 0
    assert "IS_RURAL = '1'" in d["sql"]


# ─────────────────────────────────────────────────────────────────────────────
# 15. Aggregation queries
# ─────────────────────────────────────────────────────────────────────────────

def test_members_by_district():
    d = post("total members by district")
    assert d["row_count"] >= 1
    assert_sql_has(d, "DISTRICT_NAME_ENG")


def test_gender_distribution():
    d = post("gender distribution of members")
    assert d["row_count"] >= 1
    assert_sql_has(d, "GENDER")


def test_count_by_caste_category():
    d = post("count members by caste category")
    assert d["row_count"] >= 1
    assert_sql_has(d, "CASTE_CATEGORY")


def test_avg_age_by_district():
    d = post("average age by district")
    assert d["row_count"] >= 1
    assert_sql_has(d, "AGE", "DISTRICT_NAME_ENG")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Date of birth / age queries
# ─────────────────────────────────────────────────────────────────────────────

def test_born_in_year():
    d = post("members born in 1990")
    assert d["row_count"] >= 0
    assert_sql_has(d, "DOB")


def test_born_between_years():
    d = post("members born between 1980 and 2000")
    assert d["row_count"] >= 0
    assert_sql_has(d, "DOB")


def test_age_above_60():
    d = post("show members older than 60")
    assert d["row_count"] >= 0
    assert_sql_has(d, "AGE")


# ─────────────────────────────────────────────────────────────────────────────
# 17. Combined / cross-dimensional queries
# ─────────────────────────────────────────────────────────────────────────────

def test_female_obc_rural():
    d = post("how many OBC women live in rural areas?")
    assert d["row_count"] >= 1
    assert_sql_has(d, "GENDER")


def test_sc_male_above_60():
    d = post("show SC male members above 60")
    assert d["row_count"] >= 0
    assert_sql_has(d, "CASTE_CATEGORY", "GENDER", "AGE")


def test_rural_graduates():
    d = post("how many graduates live in rural areas?")
    assert d["row_count"] >= 1
    assert "IS_RURAL = '1'" in d["sql"]
    assert_sql_has(d, "EDUCATION")


def test_urban_senior_citizens():
    d = post("senior citizens in urban areas")
    assert d["row_count"] >= 0
    assert "IS_RURAL = '0'" in d["sql"]
    assert_sql_has(d, "AGE")


def test_farmers_from_specific_district():
    d = post("show farmers from Jaipur")
    assert d["row_count"] >= 0
    assert_sql_has(d, "OCCUPATION", "DISTRICT_NAME_ENG")


# ─────────────────────────────────────────────────────────────────────────────
# 18. Typo tolerance
# ─────────────────────────────────────────────────────────────────────────────

def test_typo_jaipor():
    d = post("members from jaipor")   # typo for Jaipur
    assert d["row_count"] >= 0        # may return 0 if district doesn't match, but shouldn't crash


def test_typo_farmrs():
    d = post("how many farmrs are there?")   # typo for farmers
    assert d["row_count"] >= 0


def test_casual_phrasing():
    d = post("give me rural folks")
    assert d["row_count"] >= 0
    assert "IS_RURAL = '1'" in d["sql"]


# ─────────────────────────────────────────────────────────────────────────────
# 19. Cache integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_hit_same_result():
    """Exact repeat should cache-hit with same result."""
    q = "how many members belong to OBC caste category?"
    r1 = post(q)
    r2 = post(q)
    assert r2["source"] == "cache", "Second identical query should be cache hit"
    assert r2["row_count"] == r1["row_count"], \
        f"Cache returned different result: {r1['row_count']} vs {r2['row_count']}"


def test_cache_different_numbers_no_collision():
    """Queries with different numbers must NOT serve same cached SQL."""
    if SKIP_SLOW: return
    r1 = post("how many families have at least 1 son?")
    r2 = post("how many families have at least 2 sons?")
    # They should at minimum differ in the HAVING number
    assert r1["sql"] != r2["sql"], \
        f"Cache collision! Both queries returned identical SQL:\n{r1['sql']}"


def test_cache_semantic_different_quantifier_no_collision():
    """at-least vs more-than should NOT serve the same cached entry."""
    if SKIP_SLOW: return
    r1 = post("how many households have at least 3 children?")
    r2 = post("how many households have more than 3 children?")
    # Operator must differ (>= vs >), or at least SQL must differ
    sql1 = r1["sql"].upper()
    sql2 = r2["sql"].upper()
    assert sql1 != sql2 or ">= 3" in sql1 and "> 3" in sql2, \
        "at-least and more-than produced same SQL — cache collision or wrong operator"


# ─────────────────────────────────────────────────────────────────────────────
# 20. SQL safety (no DDL/DML)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_delete_in_output():
    d = post("delete all records")
    sql = d["sql"].upper().strip()
    assert sql.startswith("SELECT"), f"Expected SELECT, got: {sql[:60]}"


def test_no_drop_in_output():
    d = post("drop the aadhaar table")
    sql = d["sql"].upper().strip()
    assert sql.startswith("SELECT"), f"Expected SELECT, got: {sql[:60]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test registry
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # Health
    ("API health check", test_health),
    # Counts
    ("Count total members", test_count_total),
    ("Count male members", test_count_male),
    ("Count female members", test_count_female),
    ("Male + Female <= Total", test_male_plus_female_le_total),
    ("Count families", test_count_families),
    # IS_RURAL
    ("Rural → IS_RURAL = '1'", test_rural_is_1),
    ("Urban → IS_RURAL = '0'", test_urban_is_0),
    ("Village → IS_RURAL = '1'", test_village_maps_to_rural),
    ("City → IS_RURAL = '0'", test_city_maps_to_urban),
    ("Rural + Urban <= Total", test_rural_plus_urban_le_total),
    # Quantifier operators
    ("Quantifier: at least 1", test_quantifier_at_least_1),
    ("Quantifier: at least 2", test_quantifier_at_least_2),
    ("Quantifier: minimum 3", test_quantifier_minimum_3),
    ("Quantifier: more than 1", test_quantifier_more_than_1),
    ("Quantifier: more than 2", test_quantifier_more_than_2),
    ("Quantifier: greater than 3", test_quantifier_greater_than_3),
    ("Quantifier: exactly 1", test_quantifier_exactly_1),
    ("Quantifier: exactly 2", test_quantifier_exactly_2),
    ("Quantifier: at most 3", test_quantifier_at_most_3),
    ("Quantifier: no more than 2", test_quantifier_no_more_than_2),
    ("Quantifier: fewer than 3", test_quantifier_fewer_than_3),
    ("Quantifier: less than 2", test_quantifier_less_than_2),
    # Quantifier logic
    ("Logic: at_least_1 >= more_than_1", test_at_least_1_ge_more_than_1),
    ("Logic: exactly_1 <= at_least_1", test_exactly_1_subset_of_at_least_1),
    ("Logic: at_most_N <= total", test_at_most_le_total),
    # Household subquery
    ("Household: no COUNT(DISTINCT)...GROUP BY anti-pattern", test_household_count_no_count_distinct_groupby),
    ("Household: son count", test_household_son),
    ("Household: daughter count", test_household_daughter),
    ("Household: exactly 2 children", test_household_children_exact),
    # Caste
    ("Caste: OBC count", test_caste_category_obc),
    ("Caste: SC count", test_caste_category_sc),
    ("Caste: ST count", test_caste_category_st),
    ("Caste: GEN count", test_caste_category_gen),
    ("Caste: rajput", test_caste_specific_rajput),
    ("Caste: yadav", test_caste_specific_yadav),
    ("Caste: categories sum <= total", test_caste_categories_sum_le_total),
    # Age
    ("Age: senior citizens (60+)", test_senior_citizens),
    ("Age: minors under 18", test_minors),
    ("Age: adults above 18", test_adults),
    ("Age: range 20-30", test_age_range),
    ("Age: minors <= total", test_minors_le_total),
    # Education
    ("Education: graduates", test_graduates),
    ("Education: illiterates", test_illiterates),
    ("Education: graduates by district", test_education_by_district),
    # Occupation
    ("Occupation: farmers", test_farmers),
    ("Occupation: students", test_students),
    ("Occupation: unemployed", test_unemployed),
    ("Occupation: distribution", test_occupation_distribution),
    # Income
    ("Income: average", test_average_income),
    ("Income: avg by caste", test_avg_income_by_caste),
    ("Income: low income filter", test_low_income),
    ("Income: high income filter", test_high_income),
    # Banking
    ("Bank: SBI members", test_sbi_members),
    ("Bank: no bank account", test_members_with_no_bank),
    ("Bank: distribution", test_bank_distribution),
    # Marital
    ("Marital: married", test_married),
    ("Marital: unmarried", test_unmarried),
    ("Marital: widowed", test_widowed),
    # Minority
    ("Minority: count", test_minority_members),
    # Geography
    ("Geography: district filter", test_district_filter),
    ("Geography: count by district", test_district_count),
    ("Geography: block filter", test_block_filter),
    ("Geography: village filter", test_village_filter),
    # Aggregation
    ("Aggregation: members by district", test_members_by_district),
    ("Aggregation: gender distribution", test_gender_distribution),
    ("Aggregation: count by caste category", test_count_by_caste_category),
    ("Aggregation: avg age by district", test_avg_age_by_district),
    # DOB
    ("DOB: born in 1990", test_born_in_year),
    ("DOB: born between 1980-2000", test_born_between_years),
    ("DOB: age above 60", test_age_above_60),
    # Combined
    ("Combined: OBC women rural", test_female_obc_rural),
    ("Combined: SC male above 60", test_sc_male_above_60),
    ("Combined: rural graduates", test_rural_graduates),
    ("Combined: urban senior citizens", test_urban_senior_citizens),
    ("Combined: farmers from Jaipur", test_farmers_from_specific_district),
    # Typo tolerance
    ("Typo: jaipor → Jaipur", test_typo_jaipor),
    ("Typo: farmrs → farmers", test_typo_farmrs),
    ("Casual: rural folks", test_casual_phrasing),
    # Cache
    ("Cache: hit returns same result", test_cache_hit_same_result),
    ("Cache: different numbers no collision", test_cache_different_numbers_no_collision),
    ("Cache: different quantifier no collision", test_cache_semantic_different_quantifier_no_collision),
    # Safety
    ("Safety: no DELETE output", test_no_delete_in_output),
    ("Safety: no DROP output", test_no_drop_in_output),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-on-first-fail", action="store_true")
    args = parser.parse_args()
    stop_on_first_fail = args.stop_on_first_fail

    print(f"\n{'=' * 70}")
    print(f"  Jan-Aadhaar NL2SQL — Comprehensive Test Suite  ({len(TESTS)} tests)")
    print(f"  API: {API_BASE}  |  Timeout: {TIMEOUT}s per query")
    print(f"{'=' * 70}\n")

    total_t0 = time.perf_counter()
    for name, fn in TESTS:
        run_test(name, fn)

    total_ms = round((time.perf_counter() - total_t0) * 1000)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    skipped = sum(1 for r in results if r[0] == SKIP)

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"  Total time: {total_ms}ms")
    print(f"{'=' * 70}\n")

    if failed > 0:
        print("FAILED tests:")
        for r in results:
            if r[0] == FAIL:
                print(f"  * {r[1]}")
                print(f"    {r[2][:200]}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
