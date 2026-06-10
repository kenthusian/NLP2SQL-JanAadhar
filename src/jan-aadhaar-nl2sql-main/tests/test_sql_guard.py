"""
tests/test_sql_guard.py — Unit tests for the AST validation layer.

Run with:
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.sql_guard import validate

# Real column names from the Jan-Aadhaar schema
ALLOWED = [
    "DISTRICT_NAME_ENG", "IS_RURAL", "BLOCK_NAME_ENG", "CITY_NAME_ENG",
    "WARD_NAME_ENG", "GP_NAME_ENG", "VILL_NAME_ENG", "ENROLLMENT_ID",
    "MEMBER_ID", "MEM_TYPE", "RELATION_WITH_HOF", "NAME_EN",
    "FATHER_NAME_EN", "MOTHER_NAME_EN", "MARITAL_STATUS", "SPOUCE_NAME_EN",
    "DOB", "AGE", "GENDER", "CASTE_CATEGORY", "CASTE", "BANK",
    "IFSC_CODE", "ACCOUNT_NO", "MOBILE_NO", "INCOME", "OCCUPATION",
    "MINORITY", "EDUCATION",
]


# ── Valid queries ─────────────────────────────────────────────────────────────

def test_valid_count_star():
    r = validate("SELECT COUNT(*) FROM aadhaar", ALLOWED)
    assert r.ok, r.error


def test_valid_group_by():
    sql = "SELECT DISTRICT_NAME_ENG, COUNT(*) FROM aadhaar GROUP BY DISTRICT_NAME_ENG"
    r = validate(sql, ALLOWED)
    assert r.ok, r.error


def test_valid_where_filter():
    sql = "SELECT NAME_EN, AGE FROM aadhaar WHERE GENDER = 'F' AND AGE > 60 LIMIT 500"
    r = validate(sql, ALLOWED)
    assert r.ok, r.error


def test_valid_aggregation():
    sql = "SELECT CASTE_CATEGORY, AVG(INCOME) FROM aadhaar GROUP BY CASTE_CATEGORY"
    r = validate(sql, ALLOWED)
    assert r.ok, r.error


def test_valid_date_filter():
    sql = "SELECT NAME_EN, DOB FROM aadhaar WHERE DOB >= DATE '1990-01-01' LIMIT 100"
    r = validate(sql, ALLOWED)
    assert r.ok, r.error


def test_semicolon_stripped():
    r = validate("SELECT COUNT(*) FROM aadhaar;", ALLOWED)
    assert r.ok
    assert r.sql.endswith(";")   # re-appended by validator


# ── Blocked DDL / DML ─────────────────────────────────────────────────────────

def test_blocked_drop():
    r = validate("DROP TABLE aadhaar", ALLOWED)
    assert not r.ok
    assert "DROP" in r.error.upper() or "blocked" in r.error.lower()


def test_blocked_delete():
    r = validate("DELETE FROM aadhaar WHERE AGE > 80", ALLOWED)
    assert not r.ok


def test_blocked_insert():
    r = validate("INSERT INTO aadhaar (NAME_EN) VALUES ('Test')", ALLOWED)
    assert not r.ok


def test_blocked_update():
    r = validate("UPDATE aadhaar SET INCOME = 0 WHERE GENDER = 'M'", ALLOWED)
    assert not r.ok


def test_blocked_create():
    r = validate("CREATE TABLE foo AS SELECT * FROM aadhaar", ALLOWED)
    assert not r.ok


# ── Hallucinated columns ──────────────────────────────────────────────────────

def test_hallucinated_column_uid():
    r = validate("SELECT UID_HASH, NAME_EN FROM aadhaar LIMIT 10", ALLOWED)
    assert not r.ok
    assert "UID_HASH" in r.error


def test_hallucinated_column_state():
    r = validate("SELECT STATE_NAME, COUNT(*) FROM aadhaar GROUP BY STATE_NAME", ALLOWED)
    assert not r.ok


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_sql():
    r = validate("", ALLOWED)
    assert not r.ok


def test_parse_error_gibberish():
    r = validate("SELECT FROM WHERE GROUP BY ORDER HAVING", ALLOWED)
    assert not r.ok
