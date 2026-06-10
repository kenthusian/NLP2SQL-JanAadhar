"""
tests/test_rewriter.py — Deterministic tests for the SQL rewriter.

Run with:  python -m pytest tests/test_rewriter.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.sql_rewriter import rewrite, rewrite_having_operator, rewrite_is_rural


# ─────────────────────────────────────────────────────────────────────────────
# Quantifier → HAVING operator tests
# ─────────────────────────────────────────────────────────────────────────────

def _sql_having(op: str) -> str:
    return (
        f"SELECT COUNT(*) AS c FROM ("
        f"SELECT ENROLLMENT_ID FROM aadhaar GROUP BY ENROLLMENT_ID HAVING COUNT(*) {op} 2)"
    )


class TestHavingOperatorRewriter:
    """Test that HAVING COUNT(*) operators are corrected based on question phrasing."""

    def test_at_least_fixes_gt_to_gte(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "how many families have at least 2 children?")
        assert "HAVING COUNT(*) >= 2" in out

    def test_at_least_minimum(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "minimum 3 members per household")
        assert "HAVING COUNT(*) >= 3" in out or ">= 3" in out

    def test_more_than_keeps_gt(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "how many families have more than 2 sons?")
        assert "HAVING COUNT(*) > 2" in out

    def test_greater_than_keeps_gt(self):
        sql = _sql_having(">=")  # LLM wrongly used >=
        out = rewrite_having_operator(sql, "families with greater than 2 members")
        assert "HAVING COUNT(*) > 2" in out

    def test_exactly_gives_eq(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "how many households have exactly 2 daughters?")
        assert "HAVING COUNT(*) = 2" in out

    def test_just_gives_eq(self):
        sql = _sql_having(">=")
        out = rewrite_having_operator(sql, "families with just 1 son")
        assert "HAVING COUNT(*) = 1" in out

    def test_at_most_gives_lte(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "how many families have at most 3 members?")
        assert "HAVING COUNT(*) <= 3" in out

    def test_no_more_than_gives_lte(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "households with no more than 2 children")
        assert "HAVING COUNT(*) <= 2" in out

    def test_fewer_than_gives_lt(self):
        sql = _sql_having(">=")
        out = rewrite_having_operator(sql, "families with fewer than 2 sons")
        # fewer than = strict < (not <=)
        assert "HAVING COUNT(*) < 2" in out

    def test_under_gives_lt(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "households under 4 members")
        # under = strict < (not <=)
        assert "HAVING COUNT(*) < 4" in out

    def test_no_having_clause_unchanged(self):
        sql = "SELECT COUNT(*) FROM aadhaar WHERE GENDER = 'Male'"
        out = rewrite_having_operator(sql, "how many males at least 30?")
        assert out == sql  # no HAVING clause → unchanged

    def test_no_quantifier_unchanged(self):
        sql = _sql_having(">")
        out = rewrite_having_operator(sql, "how many families have sons?")
        assert out == sql  # no quantifier → unchanged


# ─────────────────────────────────────────────────────────────────────────────
# IS_RURAL rewriter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIsRuralRewriter:
    def test_r_to_1(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = 'R'"
        assert "IS_RURAL = '1'" in rewrite_is_rural(sql)

    def test_u_to_0(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = 'U'"
        assert "IS_RURAL = '0'" in rewrite_is_rural(sql)

    def test_rural_string_to_1(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = 'Rural'"
        assert "IS_RURAL = '1'" in rewrite_is_rural(sql)

    def test_urban_string_to_0(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = 'Urban'"
        assert "IS_RURAL = '0'" in rewrite_is_rural(sql)

    def test_already_correct_1_unchanged(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = '1'"
        assert rewrite_is_rural(sql) == sql

    def test_already_correct_0_unchanged(self):
        sql = "SELECT * FROM aadhaar WHERE IS_RURAL = '0'"
        assert rewrite_is_rural(sql) == sql


# ─────────────────────────────────────────────────────────────────────────────
# Integration: rewrite() applies both rewrites
# ─────────────────────────────────────────────────────────────────────────────

class TestRewriteIntegration:
    def test_full_rewrite_at_least_and_rural(self):
        sql = (
            "SELECT COUNT(*) FROM ("
            "SELECT ENROLLMENT_ID FROM aadhaar "
            "WHERE IS_RURAL = 'R' "
            "GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 1)"
        )
        out = rewrite(sql, "how many rural families have at least 1 son?")
        assert "IS_RURAL = '1'" in out
        assert "HAVING COUNT(*) >= 1" in out

    def test_full_rewrite_exactly(self):
        sql = _sql_having(">")
        out = rewrite(sql, "families with exactly 3 members")
        assert "HAVING COUNT(*) = 3" in out
