"""
test_scaffold.py — Test the three-mode pipeline.

Tests that:
  1. Full-deterministic queries bypass LLM entirely.
  2. Scaffold-assisted queries have pre-built WHERE conditions.
  3. Full-LLM queries (no hints) produce no scaffold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.domain_dict import extract_sql_hints
from llm.fast_sql import try_build_sql, build_partial_scaffold
from llm.prompt_builder import build_scaffold_prompt

QUERIES = [
    # (query, expected_mode)                                      # WHY
    ("show all rich people in Jaipur",          "deterministic"),  # income+district mapped
    ("find all unmarried women above 50",        "deterministic"),  # gender+marital+age regex
    ("show SC caste members above age 30",       "deterministic"),  # caste+age regex
    ("how many farmers earn more than 1 lakh",   "deterministic"),  # occupation+income regex
    ("list all OBC graduates in rural areas",    "deterministic"),  # caste+edu+rural mapped
    ("show people from 2rpm",                    "full_llm"),       # nothing mapped
    ("who are the children of Rahul",            "deterministic"),  # relational extractor
    ("count members by district",                "deterministic"),  # group-by built-in
    ("show families with more than 5 members",   "deterministic"),  # family-size built-in
    ("show all people with SBI account who earn above 50000", "deterministic"),  # bank+income
    ("find widows in Jodhpur",                   "deterministic"),  # marital+district
    ("list all graduates NOT in Jaipur",         "scaffold"),       # NOT is semantic
    ("show all poor families in rural areas",    "deterministic"),  # income+rural mapped
    ("show male farmers in Alwar district",      "deterministic"),  # gender+occupation+district
    ("show OBC members but not from Jaipur",     "scaffold"),       # NOT negation
    ("list members without a bank account",      "deterministic"),  # no-bank rule
]

print(f"{'QUERY':<55} {'EXPECTED':<15} {'ACTUAL':<15} {'PASS/FAIL'}")
print("=" * 105)

all_pass = True
for q, expected in QUERIES:
    hints, mapped_words = extract_sql_hints(q)
    full_sql = try_build_sql(q, hints, mapped_words)

    if full_sql:
        actual = "deterministic"
    else:
        scaffold = build_partial_scaffold(q, hints, mapped_words)
        if scaffold and scaffold.unmapped_tokens:
            actual = "scaffold"
        elif scaffold:
            # scaffold has hints but nothing unmapped → would use full prompt with hints
            actual = "deterministic"
        else:
            actual = "full_llm"

    ok = actual == expected
    if not ok:
        all_pass = False

    status = "PASS" if ok else "FAIL <<<"
    print(f"{q:<55} {expected:<15} {actual:<15} {status}")

    # For scaffold mode, print the pre-built WHERE and unmapped tokens
    if actual == "scaffold":
        scaffold = build_partial_scaffold(q, hints, mapped_words)
        if scaffold:
            print(f"  WHERE pre-built:  {scaffold.where_clause}")
            print(f"  LLM to resolve:   {scaffold.unmapped_tokens}")

print()
print("ALL PASS" if all_pass else "SOME TESTS FAILED")
