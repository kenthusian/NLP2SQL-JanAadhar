"""
Tier 3: End-to-End API Integration Tests
Tests complex combinations via the FastAPI server to ensure the LLM
generates valid DuckDB SQL that passes through sqlglot and execute correctly.
"""
import requests
import sys

PASS = "[PASS]"
FAIL = "[FAIL]"

URL = "http://127.0.0.1:8000/query"

# Test cases: (Question, Expected fragments in SQL)
TESTS = [
    (
        "Show people from Srinagar and Beejasar",
        ["Srinagar", "Beejasar", "OR"]
    ),
    (
        "How many families have more than 5 members?",
        ["GROUP BY", "HAVING", "> 5"]
    ),
    (
        "Who has SBI or PNB account?",
        ["State Bank of India", "Punjab National Bank"]
    ),
    (
        "Show members who do not have a bank account",
        ["ACCOUNT_NO", "IS NULL"]
    ),
    (
        "Who are the daughters of Priya?",
        ["MOTHER_NAME_EN", "Priya"]
    ),
    # Hard combinations
    (
        "How many families in Srinagar or Beejasar have more than 2 members?",
        ["Srinagar", "Beejasar", "HAVING", "> 2"]
    ),
    (
        "Which unmarried women do not have a bank account?",
        ["Unmarried", "Female", "IS NULL"]
    )
]

def run_tests():
    passed = 0
    failed = 0
    
    for q, expected_fragments in TESTS:
        print(f"\nTesting: {q!r}")
        try:
            resp = requests.post(URL, json={"question": q}, timeout=300)
            if resp.status_code != 200:
                print(f"{FAIL} HTTP {resp.status_code}")
                failed += 1
                continue
                
            data = resp.json()
            if data.get("status") != "success":
                print(f"{FAIL} API returned status: {data.get('status')}")
                print(f"       Error: {data.get('error')}")
                failed += 1
                continue
                
            sql = data.get("sql", "").upper()
            missing = [f for f in expected_fragments if f.upper() not in sql]
            
            if missing:
                print(f"{FAIL} Missing expected SQL fragments: {missing}")
                print(f"       Generated SQL: {sql}")
                failed += 1
            else:
                print(f"{PASS} Valid query execution.")
                passed += 1
                
        except Exception as e:
            print(f"{FAIL} Exception: {e}")
            failed += 1
            
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*40}")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
