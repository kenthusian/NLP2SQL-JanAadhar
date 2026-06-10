from app import _post_process_sql, _get_cache
from validation.sql_validator import SQLValidator
from optimization.query_optimizer import QueryOptimizer
from llm.fast_path import FastPathEngine

print("=== Testing AST SQL Post-Processing ===")
post_process_tests = [
    ("SELECT * FROM member WHERE gender = 'boy'", "SELECT * FROM citizen WHERE gender = 'Male';"),
    ("SELECT citizen.age FROM family WHERE caste_category = 'General'", "SELECT age FROM citizen WHERE caste_category = 'GEN';"),
    ("SELECT * FROM citizen WHERE is_rural = 'rural'", "SELECT * FROM citizen WHERE is_rural = 1;"),
    ("SELECT * FROM citizen WHERE bank_name = 'SBI'", "SELECT * FROM citizen WHERE UPPER(bank_name) LIKE '%SBI%';"),
]

passed_ast = 0
for sql_in, expected in post_process_tests:
    res = _post_process_sql(sql_in)
    if res == expected:
        passed_ast += 1
        print(f"[OK] {sql_in}")
    else:
        print(f"[FAIL] {sql_in}\n  Expected: {expected}\n  Got:      {res}")
        
print(f"{passed_ast}/{len(post_process_tests)} AST tests passed\n")


print("=== Testing SQL Validator ===")
v = SQLValidator()
validator_tests = [
    ("SELECT COUNT(*) FROM citizen", True),
    ("DROP TABLE citizen", False),
    ("SELECT * FROM citizen JOIN results_abc ON 1=1", True),
    ("SELECT * FROM citizen JOIN family ON 1=1", False),
    ("SELECT fake_col FROM citizen", False),
    ("SELECT age FROM citizen", True),
]

passed_val = 0
for sql_in, expected_valid in validator_tests:
    allowed_t = ["citizen", "results_abc"] if "results_abc" in sql_in else None
    res = v.validate(sql_in, allowed_tables=allowed_t)
    if res.valid == expected_valid:
        passed_val += 1
        print(f"[OK] {sql_in}")
    else:
        print(f"[FAIL] {sql_in} (Expected {expected_valid}, got {res.valid}: {res.errors})")
        
print(f"{passed_val}/{len(validator_tests)} Validator tests passed\n")

print("=== Testing Fast Path Engine ===")
fast = FastPathEngine()
simple_queries = [
    ("Show me males in Jaipur", True),
    ("SC widows above 60", True),
    ("Citizens with income above 2 lakhs", True),
    ("Average income of SC citizens", False),
    ("How many farmers in each district", False),
    ("Show me the data of people whose height is above 5ft", False),
]

passed_fast = 0
for q, expected in simple_queries:
    sql = fast.generate_sql_fast(q)
    success = (sql is not None)
    if success == expected:
        passed_fast += 1
        print(f"[OK] {q} -> {sql}")
    else:
        print(f"[FAIL] {q} Expected {expected}, got {success}: {sql}")
print(f"{passed_fast}/{len(simple_queries)} Fast Path tests passed\n")


print("=== Testing AST Swapping ===")
cached_q = "Show me males in Jaipur"
cached_sql = "SELECT member_name, age, gender, district FROM citizen WHERE gender = 'Male' AND district = 'JAIPUR';"

swap_tests = [
    ("Show me females in Jaipur", True, "SELECT member_name, age, gender, district FROM citizen WHERE gender = 'Female' AND district = 'JAIPUR';"),
    ("Show me males in Jodhpur", True, "SELECT member_name, age, gender, district FROM citizen WHERE gender = 'Male' AND district = 'JODHPUR';"),
    ("Show me boys in Jaipur", False, None),
    ("Show me females in Jodhpur", True, "SELECT member_name, age, gender, district FROM citizen WHERE gender = 'Female' AND district = 'JODHPUR';"),
]

passed_swap = 0
for new_q, expected, expected_sql in swap_tests:
    sql = fast.swap_ast_parameters(cached_sql, cached_q, new_q)
    success = (sql is not None)
    if success == expected and (not expected or sql == expected_sql):
        passed_swap += 1
        print(f"[OK] {new_q} -> {sql}")
    else:
        print(f"[FAIL] {new_q}\n  Expected: {expected_sql}\n  Got:      {sql}")

print(f"{passed_swap}/{len(swap_tests)} Swap tests passed\n")

