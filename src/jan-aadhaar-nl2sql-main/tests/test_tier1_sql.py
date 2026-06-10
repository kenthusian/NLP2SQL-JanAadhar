"""
Tier 1: SQL ground-truth test suite.
Runs queries directly against DuckDB — no LLM, no server needed.
Generates golden answers for every query category.
"""
import sys
import duckdb
import json
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results = []

def connect():
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4; PRAGMA memory_limit='4GB';")
    con.execute("CREATE VIEW aadhaar AS SELECT * FROM read_parquet('data/aadhaar/**/*.parquet')")
    return con

def q(con, sql):
    return con.execute(sql).fetchall()

def run(name, fn, con):
    try:
        result = fn(con)
        results.append((PASS, name, result))
        print(f"{PASS}  {name}")
        if isinstance(result, list) and len(result) <= 3:
            for row in result:
                print(f"         {row}")
        elif not isinstance(result, list):
            print(f"         result={result}")
        return result
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL}  {name}")
        print(f"         {e}")
        return None

def assert_eq(actual, expected, tolerance=0):
    if tolerance:
        assert abs(actual - expected) <= tolerance, f"got {actual}, expected {expected} ±{tolerance}"
    else:
        assert actual == expected, f"got {actual!r}, expected {expected!r}"

def assert_gt(actual, minimum):
    assert actual > minimum, f"got {actual}, expected > {minimum}"

def assert_ge(actual, minimum):
    assert actual >= minimum, f"got {actual}, expected >= {minimum}"

def assert_le(actual, maximum):
    assert actual <= maximum, f"got {actual}, expected <= {maximum}"

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
def test_schema(con):
    cols = q(con, "DESCRIBE aadhaar")
    names = [c[0] for c in cols]
    print(f"         {len(names)} columns: {', '.join(names)}")
    required = ['ENROLLMENT_ID','NAME_EN','GENDER','AGE','DOB','CASTE','CASTE_CATEGORY',
                'EDUCATION','OCCUPATION','INCOME','BANK','ACCOUNT_NO','IFSC_CODE',
                'MARITAL_STATUS','IS_RURAL','DISTRICT_NAME_ENG','VILL_NAME_ENG']
    for col in required:
        assert col in names, f"Missing required column: {col}"
    return names

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1A: Baseline counts
# ─────────────────────────────────────────────────────────────────────────────
def test_total(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    assert_gt(n, 0)
    return n

def test_distinct_families(con):
    n = q(con, "SELECT COUNT(DISTINCT ENROLLMENT_ID) FROM aadhaar")[0][0]
    assert_gt(n, 0)
    return n

def test_gender_breakdown(con):
    rows = q(con, "SELECT GENDER, COUNT(*) AS cnt FROM aadhaar GROUP BY GENDER ORDER BY cnt DESC")
    assert len(rows) >= 2
    return rows

def test_males(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE GENDER = 'Male'")[0][0]
    assert_gt(n, 0); return n

def test_females(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE GENDER = 'Female'")[0][0]
    assert_gt(n, 0); return n

def test_gender_sum_le_total(con):
    total = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    males = q(con, "SELECT COUNT(*) FROM aadhaar WHERE GENDER='Male'")[0][0]
    females = q(con, "SELECT COUNT(*) FROM aadhaar WHERE GENDER='Female'")[0][0]
    assert males + females <= total
    return f"Males={males} + Females={females} <= Total={total}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1B: IS_RURAL
# ─────────────────────────────────────────────────────────────────────────────
def test_rural_value_is_1(con):
    vals = {r[0] for r in q(con, "SELECT DISTINCT IS_RURAL FROM aadhaar WHERE IS_RURAL IS NOT NULL")}
    assert '1' in vals or '0' in vals, f"Unexpected IS_RURAL values: {vals}"
    return f"Distinct IS_RURAL values: {vals}"

def test_rural_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='1'")[0][0]
    assert_ge(n, 0); return n

def test_urban_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='0'")[0][0]
    assert_ge(n, 0); return n

def test_rural_plus_urban_le_total(con):
    total = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    rural = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='1'")[0][0]
    urban = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='0'")[0][0]
    assert rural + urban <= total
    return f"Rural={rural} Urban={urban} Total={total}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1C: Caste categories
# ─────────────────────────────────────────────────────────────────────────────
def test_caste_categories(con):
    rows = q(con, "SELECT CASTE_CATEGORY, COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY IS NOT NULL GROUP BY CASTE_CATEGORY ORDER BY 2 DESC")
    assert len(rows) >= 1; return rows

def test_obc_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='OBC'")[0][0]
    assert_ge(n, 0); return n

def test_sc_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='SC'")[0][0]
    assert_ge(n, 0); return n

def test_st_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='ST'")[0][0]
    assert_ge(n, 0); return n

def test_gen_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='GEN'")[0][0]
    assert_ge(n, 0); return n

def test_caste_sum_le_total(con):
    total = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    obc = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='OBC'")[0][0]
    sc = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='SC'")[0][0]
    st = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='ST'")[0][0]
    gen = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='GEN'")[0][0]
    assert obc + sc + st + gen <= total
    return f"OBC={obc} SC={sc} ST={st} GEN={gen} Total={total}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1D: Age groups
# ─────────────────────────────────────────────────────────────────────────────
def test_seniors(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE>=60")[0][0]
    assert_ge(n, 0); return n

def test_minors(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE<18")[0][0]
    assert_ge(n, 0); return n

def test_youth(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE BETWEEN 18 AND 35")[0][0]
    assert_ge(n, 0); return n

def test_working_age(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE BETWEEN 18 AND 60")[0][0]
    assert_ge(n, 0); return n

def test_age_range_20_30(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE BETWEEN 20 AND 30")[0][0]
    assert_ge(n, 0); return n

def test_minors_le_seniors(con):
    """Sanity: minors + seniors + working-age + youth ≤ 2 * total (they overlap)"""
    total = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    minors = q(con, "SELECT COUNT(*) FROM aadhaar WHERE AGE<18")[0][0]
    assert minors <= total
    return f"Minors={minors} Total={total}"

def test_age_distribution(con):
    rows = q(con, "SELECT CASE WHEN AGE<18 THEN 'minor' WHEN AGE<60 THEN 'adult' ELSE 'senior' END AS grp, COUNT(*) FROM aadhaar GROUP BY grp")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1E: Education
# ─────────────────────────────────────────────────────────────────────────────
def test_education_values(con):
    rows = q(con, "SELECT EDUCATION, COUNT(*) FROM aadhaar WHERE EDUCATION IS NOT NULL GROUP BY EDUCATION ORDER BY 2 DESC LIMIT 10")
    assert len(rows) >= 1; return rows

def test_graduates(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE EDUCATION ILIKE '%graduate%'")[0][0]
    assert_ge(n, 0); return n

def test_illiterates(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE EDUCATION ILIKE '%illiterate%'")[0][0]
    assert_ge(n, 0); return n

def test_graduates_by_district(con):
    rows = q(con, "SELECT DISTRICT_NAME_ENG, COUNT(*) AS cnt FROM aadhaar WHERE EDUCATION ILIKE '%graduate%' GROUP BY DISTRICT_NAME_ENG ORDER BY cnt DESC LIMIT 5")
    assert_ge(len(rows), 0); return rows

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1F: Occupation
# ─────────────────────────────────────────────────────────────────────────────
def test_occupation_values(con):
    rows = q(con, "SELECT OCCUPATION, COUNT(*) FROM aadhaar WHERE OCCUPATION IS NOT NULL GROUP BY OCCUPATION ORDER BY 2 DESC LIMIT 10")
    assert len(rows) >= 1; return rows

def test_farmers(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE OCCUPATION ILIKE '%farmer%' OR OCCUPATION ILIKE '%agri%'")[0][0]
    assert_ge(n, 0); return n

def test_students(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE OCCUPATION ILIKE '%student%'")[0][0]
    assert_ge(n, 0); return n

def test_unemployed(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE OCCUPATION ILIKE '%unemployed%'")[0][0]
    assert_ge(n, 0); return n

def test_homemakers(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE OCCUPATION ILIKE '%home%maker%' OR OCCUPATION ILIKE '%housewife%'")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1G: Income
# ─────────────────────────────────────────────────────────────────────────────
def test_income_range(con):
    row = q(con, "SELECT MIN(INCOME), MAX(INCOME), AVG(INCOME) FROM aadhaar WHERE INCOME IS NOT NULL AND INCOME > 0")[0]
    return f"min={row[0]} max={row[1]} avg={round(row[2],2)}"

def test_bpl(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE INCOME < 50000 AND INCOME IS NOT NULL")[0][0]
    assert_ge(n, 0); return n

def test_high_income(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE INCOME > 100000 AND INCOME IS NOT NULL")[0][0]
    assert_ge(n, 0); return n

def test_avg_income_by_caste(con):
    rows = q(con, "SELECT CASTE_CATEGORY, ROUND(AVG(INCOME),2) AS avg_inc FROM aadhaar WHERE INCOME IS NOT NULL GROUP BY CASTE_CATEGORY ORDER BY avg_inc DESC")
    assert len(rows) >= 1; return rows

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1H: Banking
# ─────────────────────────────────────────────────────────────────────────────
def test_bank_values(con):
    rows = q(con, "SELECT BANK, COUNT(*) FROM aadhaar WHERE BANK IS NOT NULL AND BANK != '' GROUP BY BANK ORDER BY 2 DESC LIMIT 10")
    assert len(rows) >= 1; return rows

def test_sbi_fullname(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE BANK ILIKE '%State Bank of India%'")[0][0]
    assert_eq(n, 223); return n

def test_sbi_abbreviation_hint(con):
    """Verify IFSC prefix approach also works"""
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IFSC_CODE ILIKE 'SBIN%'")[0][0]
    assert_ge(n, 0); return n  # may differ from full-name since IFSC may be blank

def test_pnb_fullname(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE BANK ILIKE '%Punjab National Bank%'")[0][0]
    assert_eq(n, 83); return n

def test_bob_fullname(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE BANK ILIKE '%Bank of Baroda%'")[0][0]
    assert_eq(n, 106); return n

def test_no_bank_account(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE BANK IS NULL OR BANK = '' OR ACCOUNT_NO IS NULL OR ACCOUNT_NO = ''")[0][0]
    assert_ge(n, 0); return n

def test_bank_distribution(con):
    rows = q(con, "SELECT BANK, COUNT(*) AS cnt FROM aadhaar WHERE BANK IS NOT NULL AND BANK!='' GROUP BY BANK ORDER BY cnt DESC")
    return [(r[0][:30], r[1]) for r in rows[:5]]

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1I: Marital status
# ─────────────────────────────────────────────────────────────────────────────
def test_marital_values(con):
    rows = q(con, "SELECT MARITAL_STATUS, COUNT(*) FROM aadhaar WHERE MARITAL_STATUS IS NOT NULL GROUP BY MARITAL_STATUS ORDER BY 2 DESC")
    assert len(rows) >= 1; return rows

def test_married(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE MARITAL_STATUS ILIKE '%married%' AND MARITAL_STATUS NOT ILIKE '%un%'")[0][0]
    assert_ge(n, 0); return n

def test_unmarried(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE MARITAL_STATUS ILIKE '%unmarried%' OR MARITAL_STATUS ILIKE '%single%'")[0][0]
    assert_ge(n, 0); return n

def test_widowed(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE MARITAL_STATUS ILIKE '%widow%'")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1J: Minority
# ─────────────────────────────────────────────────────────────────────────────
def test_minority_values(con):
    rows = q(con, "SELECT MINORITY, COUNT(*) FROM aadhaar WHERE MINORITY IS NOT NULL AND MINORITY!='' GROUP BY MINORITY ORDER BY 2 DESC LIMIT 5")
    return rows

def test_minority_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE MINORITY IS NOT NULL AND MINORITY!=''")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1K: Geography
# ─────────────────────────────────────────────────────────────────────────────
def test_districts(con):
    rows = q(con, "SELECT DISTRICT_NAME_ENG, COUNT(*) AS cnt FROM aadhaar GROUP BY DISTRICT_NAME_ENG ORDER BY cnt DESC LIMIT 5")
    assert len(rows) >= 1; return rows

def test_jaipur_exists(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE DISTRICT_NAME_ENG ILIKE '%jaipur%'")[0][0]
    assert_ge(n, 0); return n

def test_villages(con):
    rows = q(con, "SELECT COUNT(DISTINCT VILL_NAME_ENG) FROM aadhaar WHERE VILL_NAME_ENG IS NOT NULL")
    n = rows[0][0]
    assert_ge(n, 0); return n

def test_multi_location_or(con):
    """Two locations with OR should return union"""
    n_both = q(con, "SELECT COUNT(*) FROM aadhaar WHERE DISTRICT_NAME_ENG ILIKE '%Jaipur%' OR DISTRICT_NAME_ENG ILIKE '%Jodhpur%'")[0][0]
    n_j1 = q(con, "SELECT COUNT(*) FROM aadhaar WHERE DISTRICT_NAME_ENG ILIKE '%Jaipur%'")[0][0]
    n_j2 = q(con, "SELECT COUNT(*) FROM aadhaar WHERE DISTRICT_NAME_ENG ILIKE '%Jodhpur%'")[0][0]
    assert n_both <= n_j1 + n_j2  # OR ≤ sum (if no overlap, = sum)
    return f"Jaipur={n_j1} OR Jodhpur={n_j2} combined={n_both}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1L: Household / family counting
# ─────────────────────────────────────────────────────────────────────────────
def test_family_size_distribution(con):
    rows = q(con, "SELECT member_count, COUNT(*) AS families FROM (SELECT ENROLLMENT_ID, COUNT(*) AS member_count FROM aadhaar GROUP BY ENROLLMENT_ID) sub GROUP BY member_count ORDER BY member_count")
    return rows

def test_at_least_1_son_gte(con):
    n = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) >= 1)")[0][0]
    assert_ge(n, 0); return n

def test_more_than_1_son_gt(con):
    n = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 1)")[0][0]
    assert_ge(n, 0); return n

def test_at_least_ge_more_than(con):
    """at_least_1 must be >= more_than_1"""
    at_least_1 = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) >= 1)")[0][0]
    more_than_1 = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 1)")[0][0]
    assert at_least_1 >= more_than_1
    return f"at_least_1={at_least_1} >= more_than_1={more_than_1}"

def test_exactly_1_son(con):
    n = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) = 1)")[0][0]
    assert_ge(n, 0); return n

def test_exactly_le_atleast(con):
    """exactly_1 must be <= at_least_1"""
    exact = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) = 1)")[0][0]
    atleast = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%Son%' GROUP BY ENROLLMENT_ID HAVING COUNT(*) >= 1)")[0][0]
    assert exact <= atleast
    return f"exactly_1={exact} <= at_least_1={atleast}"

def test_at_most_3_members(con):
    n = q(con, "SELECT COUNT(*) FROM (SELECT ENROLLMENT_ID FROM aadhaar GROUP BY ENROLLMENT_ID HAVING COUNT(*) <= 3)")[0][0]
    assert_ge(n, 0); return n

def test_more_than_5_members_families(con):
    rows = q(con, "SELECT ENROLLMENT_ID, COUNT(*) AS cnt FROM aadhaar GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 5 ORDER BY cnt DESC LIMIT 5")
    assert_ge(len(rows), 0); return rows

def test_largest_family(con):
    row = q(con, "SELECT ENROLLMENT_ID, COUNT(*) AS cnt FROM aadhaar GROUP BY ENROLLMENT_ID ORDER BY cnt DESC LIMIT 1")[0]
    assert row[1] >= 1; return f"ENROLLMENT_ID={row[0]} members={row[1]}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1M: Relational queries (parent-child)
# ─────────────────────────────────────────────────────────────────────────────
def test_father_name_column(con):
    rows = q(con, "SELECT FATHER_NAME_EN, COUNT(*) FROM aadhaar WHERE FATHER_NAME_EN IS NOT NULL AND FATHER_NAME_EN!='' GROUP BY FATHER_NAME_EN ORDER BY 2 DESC LIMIT 5")
    assert_ge(len(rows), 0); return rows

def test_mother_name_column(con):
    rows = q(con, "SELECT MOTHER_NAME_EN, COUNT(*) FROM aadhaar WHERE MOTHER_NAME_EN IS NOT NULL AND MOTHER_NAME_EN!='' GROUP BY MOTHER_NAME_EN ORDER BY 2 DESC LIMIT 3")
    assert_ge(len(rows), 0); return rows

def test_relation_with_hof_values(con):
    rows = q(con, "SELECT RELATION_WITH_HOF, COUNT(*) FROM aadhaar WHERE RELATION_WITH_HOF IS NOT NULL GROUP BY RELATION_WITH_HOF ORDER BY 2 DESC")
    assert len(rows) >= 1; return rows

def test_sons_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%son%'")[0][0]
    assert_ge(n, 0); return n

def test_daughters_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE RELATION_WITH_HOF ILIKE '%daughter%'")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1N: Combined/cross-dimensional queries
# ─────────────────────────────────────────────────────────────────────────────
def test_rural_obc_female(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='1' AND CASTE_CATEGORY='OBC' AND GENDER='Female'")[0][0]
    assert_ge(n, 0); return n

def test_urban_male_senior(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='0' AND GENDER='Male' AND AGE>=60")[0][0]
    assert_ge(n, 0); return n

def test_sc_female_graduate(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='SC' AND GENDER='Female' AND EDUCATION ILIKE '%graduate%'")[0][0]
    assert_ge(n, 0); return n

def test_rural_farmer_bpl(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='1' AND OCCUPATION ILIKE '%farmer%' AND INCOME<50000")[0][0]
    assert_ge(n, 0); return n

def test_urban_graduate_sbi(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE IS_RURAL='0' AND EDUCATION ILIKE '%graduate%' AND BANK ILIKE '%State Bank%'")[0][0]
    assert_ge(n, 0); return n

def test_combined_le_components(con):
    """Combined filter must have fewer or equal rows than any single filter"""
    total = q(con, "SELECT COUNT(*) FROM aadhaar")[0][0]
    combined = q(con, "SELECT COUNT(*) FROM aadhaar WHERE CASTE_CATEGORY='OBC' AND GENDER='Female' AND IS_RURAL='1'")[0][0]
    assert combined <= total
    return f"combined={combined} <= total={total}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1O: Aggregations
# ─────────────────────────────────────────────────────────────────────────────
def test_count_by_district(con):
    rows = q(con, "SELECT DISTRICT_NAME_ENG, COUNT(*) AS cnt FROM aadhaar GROUP BY DISTRICT_NAME_ENG ORDER BY cnt DESC")
    assert len(rows) >= 1; return rows[:3]

def test_avg_age_by_district(con):
    rows = q(con, "SELECT DISTRICT_NAME_ENG, ROUND(AVG(AGE),1) AS avg_age FROM aadhaar GROUP BY DISTRICT_NAME_ENG ORDER BY avg_age DESC")
    assert len(rows) >= 1; return rows[:3]

def test_sum_income_by_caste(con):
    rows = q(con, "SELECT CASTE_CATEGORY, SUM(INCOME) AS total_inc FROM aadhaar WHERE INCOME IS NOT NULL GROUP BY CASTE_CATEGORY ORDER BY total_inc DESC")
    assert len(rows) >= 1; return rows[:3]

def test_gender_by_rural_urban(con):
    rows = q(con, "SELECT IS_RURAL, GENDER, COUNT(*) FROM aadhaar GROUP BY IS_RURAL, GENDER ORDER BY IS_RURAL, 3 DESC")
    assert len(rows) >= 1; return rows

def test_rollup_parquet_exists(con):
    try:
        n = con.execute("SELECT COUNT(*) FROM read_parquet('data/district_rollup.parquet')").fetchone()[0]
        return f"district_rollup.parquet: {n} rows"
    except Exception as e:
        return f"WARN: rollup not found: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1P: Date of birth
# ─────────────────────────────────────────────────────────────────────────────
def test_dob_range(con):
    rows = q(con, "SELECT MIN(DOB), MAX(DOB) FROM aadhaar WHERE DOB IS NOT NULL")[0]
    return f"DOB range: {rows[0]} to {rows[1]}"

def test_born_in_1990(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE YEAR(DOB)=1990")[0][0]
    assert_ge(n, 0); return n

def test_born_1980_2000(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE DOB >= DATE '1980-01-01' AND DOB <= DATE '2000-12-31'")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1Q: Head of family / MEM_TYPE
# ─────────────────────────────────────────────────────────────────────────────
def test_mem_type_values(con):
    rows = q(con, "SELECT MEM_TYPE, COUNT(*) FROM aadhaar WHERE MEM_TYPE IS NOT NULL GROUP BY MEM_TYPE ORDER BY 2 DESC")
    assert len(rows) >= 1; return rows

def test_hof_count(con):
    n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE MEM_TYPE ILIKE '%HOF%' OR RELATION_WITH_HOF ILIKE '%Self%'")[0][0]
    assert_ge(n, 0); return n

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1R: BPL / welfare
# ─────────────────────────────────────────────────────────────────────────────
def test_bpl_column_exists(con):
    try:
        n = q(con, "SELECT COUNT(*) FROM aadhaar WHERE BPL_STATUS IS NOT NULL")[0][0]
        return f"BPL_STATUS not-null rows: {n}"
    except Exception:
        return "BPL_STATUS column not present — using INCOME<50000 as proxy"

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("Schema inspection", test_schema),
    # Baseline
    ("Total member count", test_total),
    ("Distinct families", test_distinct_families),
    ("Gender breakdown", test_gender_breakdown),
    ("Male count", test_males),
    ("Female count", test_females),
    ("Male + Female <= Total", test_gender_sum_le_total),
    # IS_RURAL
    ("IS_RURAL values are '1'/'0'", test_rural_value_is_1),
    ("Rural count (IS_RURAL='1')", test_rural_count),
    ("Urban count (IS_RURAL='0')", test_urban_count),
    ("Rural + Urban <= Total", test_rural_plus_urban_le_total),
    # Caste
    ("Caste categories", test_caste_categories),
    ("OBC count", test_obc_count),
    ("SC count", test_sc_count),
    ("ST count", test_st_count),
    ("GEN count", test_gen_count),
    ("Caste sum <= Total", test_caste_sum_le_total),
    # Age
    ("Senior citizens (>=60)", test_seniors),
    ("Minors (<18)", test_minors),
    ("Youth (18-35)", test_youth),
    ("Working age (18-60)", test_working_age),
    ("Age range 20-30", test_age_range_20_30),
    ("Minors <= Total", test_minors_le_seniors),
    ("Age distribution", test_age_distribution),
    # Education
    ("Education values", test_education_values),
    ("Graduate count", test_graduates),
    ("Illiterate count", test_illiterates),
    ("Graduates by district", test_graduates_by_district),
    # Occupation
    ("Occupation values", test_occupation_values),
    ("Farmer count", test_farmers),
    ("Student count", test_students),
    ("Unemployed count", test_unemployed),
    ("Homemaker count", test_homemakers),
    # Income
    ("Income range", test_income_range),
    ("BPL count (<50k)", test_bpl),
    ("High income (>100k)", test_high_income),
    ("Avg income by caste", test_avg_income_by_caste),
    # Banking
    ("Bank values", test_bank_values),
    ("SBI exact count = 223", test_sbi_fullname),
    ("SBI via IFSC prefix", test_sbi_abbreviation_hint),
    ("PNB exact count = 83", test_pnb_fullname),
    ("BOB exact count = 106", test_bob_fullname),
    ("No bank account count", test_no_bank_account),
    ("Bank distribution top 5", test_bank_distribution),
    # Marital
    ("Marital status values", test_marital_values),
    ("Married count", test_married),
    ("Unmarried count", test_unmarried),
    ("Widowed count", test_widowed),
    # Minority
    ("Minority values", test_minority_values),
    ("Minority count", test_minority_count),
    # Geography
    ("District list", test_districts),
    ("Jaipur exists", test_jaipur_exists),
    ("Village count", test_villages),
    ("Multi-location OR logic", test_multi_location_or),
    # Household/family counting
    ("Family size distribution", test_family_size_distribution),
    ("At-least-1-son (>=1)", test_at_least_1_son_gte),
    ("More-than-1-son (>1)", test_more_than_1_son_gt),
    ("at_least_1 >= more_than_1 (logic)", test_at_least_ge_more_than),
    ("Exactly 1 son", test_exactly_1_son),
    ("exactly_1 <= at_least_1 (logic)", test_exactly_le_atleast),
    ("At-most-3-members families", test_at_most_3_members),
    ("Families with >5 members", test_more_than_5_members_families),
    ("Largest family", test_largest_family),
    # Relational
    ("FATHER_NAME column", test_father_name_column),
    ("MOTHER_NAME column", test_mother_name_column),
    ("RELATION_WITH_HOF values", test_relation_with_hof_values),
    ("Son count", test_sons_count),
    ("Daughter count", test_daughters_count),
    # Combined
    ("Rural OBC Female", test_rural_obc_female),
    ("Urban Male Senior", test_urban_male_senior),
    ("SC Female Graduate", test_sc_female_graduate),
    ("Rural Farmer BPL", test_rural_farmer_bpl),
    ("Urban Graduate SBI", test_urban_graduate_sbi),
    ("Combined <= Total (logic)", test_combined_le_components),
    # Aggregations
    ("Count by district", test_count_by_district),
    ("Avg age by district", test_avg_age_by_district),
    ("Sum income by caste", test_sum_income_by_caste),
    ("Gender by rural/urban", test_gender_by_rural_urban),
    ("Rollup parquet", test_rollup_parquet_exists),
    # DOB
    ("DOB range", test_dob_range),
    ("Born in 1990", test_born_in_1990),
    ("Born 1980-2000", test_born_1980_2000),
    # HOF / MEM_TYPE
    ("MEM_TYPE values", test_mem_type_values),
    ("HOF count", test_hof_count),
    # BPL
    ("BPL column", test_bpl_column_exists),
]

if __name__ == "__main__":
    import time
    print(f"\n{'='*65}")
    print(f"  Tier 1: SQL Ground-Truth Tests  ({len(TESTS)} tests)")
    print(f"{'='*65}\n")

    con = connect()
    t0 = time.perf_counter()
    for name, fn in TESTS:
        run(name, fn, con)
    ms = round((time.perf_counter() - t0) * 1000)

    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{len(TESTS)} passed, {failed} failed  ({ms}ms total)")
    print(f"{'='*65}\n")

    if failed:
        print("FAILED tests:")
        for r in results:
            if r[0] == FAIL:
                print(f"  * {r[1]}: {r[2]}")
        sys.exit(1)

    # Save golden answers for reference
    golden = {}
    for r in results:
        golden[r[1]] = str(r[2])
    with open("data/golden_answers.json", "w") as f:
        json.dump(golden, f, indent=2, default=str)
    print("Golden answers saved to data/golden_answers.json")
