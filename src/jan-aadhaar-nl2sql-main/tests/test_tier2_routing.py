"""
Tier 2: Domain Routing & Extraction Tests
Verifies that natural language questions properly trigger the correct SQL hints
in `rag/domain_dict.py`. Runs instantly without LLM or DB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.domain_dict import extract_sql_hints

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def check(question: str, expected_fragment: str):
    hints = extract_sql_hints(question)
    combined = " | ".join(hints).upper()
    
    if expected_fragment.upper() in combined:
        results.append((PASS, question))
        print(f"{PASS}  {question!r}")
    else:
        results.append((FAIL, question, f"Expected {expected_fragment!r} not in {hints}"))
        print(f"{FAIL}  {question!r}")
        print(f"         Got: {hints}")

def run_tests():
    # 1. Banks (Issue 1)
    check("show members with SBI account", "SBIN")
    check("members with PNB bank", "PUNB")
    check("who has HDFC account", "HDFC")
    check("show BOB members", "BARB")
    check("who has no bank account", "IS NULL")
    check("unbanked members", "IS NULL")
    
    # 2. Income
    check("poor families", "INCOME < 50000")
    check("low income people", "INCOME < 50000")
    check("below poverty line", "INCOME < 50000")
    check("rich people", "INCOME > 500000")
    check("zero income members", "INCOME = 0")
    
    # 3. Education
    check("illiterate people", "illiterate")
    check("graduates", "graduate")
    check("10th pass", "10")
    check("12th pass", "12")
    check("postgraduate members", "post")
    
    # 4. Age
    check("senior citizens", "AGE >= 60")
    check("minors", "AGE < 18")
    check("youth", "18 AND 35")
    check("working age", "18 AND 60")
    
    # 5. Marital
    check("widows", "Widow")
    check("single men", "Unmarried")
    check("divorced women", "Divorced")
    
    # 6. Occupation
    check("farmers", "farmer")
    check("daily wage labourers", "labour")
    check("housewives", "housewife")
    
    # 7. Minority
    check("minority communities", "MINORITY IS NOT NULL")
    check("muslims", "MINORITY IS NOT NULL")
    
    # 8. Caste (Fuzzy matching)
    check("show jaat people", "28 JAT")
    check("show me data of brahmins", "BRAHMIN")
    check("list all rajputs", "RAJPUT")
    
if __name__ == "__main__":
    print(f"\n{'='*65}")
    print(f"  Tier 2: Domain Routing Tests")
    print(f"{'='*65}\n")
    
    run_tests()
    
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    
    print(f"\n{'='*65}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*65}\n")
    
    if failed:
        sys.exit(1)
