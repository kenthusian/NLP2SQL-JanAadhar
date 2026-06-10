from app import _post_process_sql

tests = [
    (
        "SELECT member_name, age, gender, district, ward FROM citizen WHERE district = 'Jaipur' AND ward = '62';",
        "TRIM(ward) LIKE '%62%'",
    ),
    (
        "SELECT * FROM citizen WHERE city = 'Sanganer';",
        "TRIM(city) LIKE '%Sanganer%'",
    ),
    (
        "SELECT * FROM citizen WHERE block = 'Phulera';",
        "TRIM(block) LIKE '%Phulera%'",
    ),
    (
        "SELECT * FROM citizen WHERE village = 'Kalwar';",
        "TRIM(village) LIKE '%Kalwar%'",
    ),
    # Should NOT change — ward already using LIKE
    (
        "SELECT * FROM citizen WHERE ward LIKE '%62%';",
        "ward LIKE '%62%'",
    ),
]

passed = 0
for sql, expected_fragment in tests:
    result = _post_process_sql(sql)
    ok = expected_fragment in result
    print(f"[{'OK' if ok else 'FAIL'}] {sql[:60]}...")
    print(f"         => {result}")
    if not ok:
        print(f"         Expected fragment: {expected_fragment}")
    print()
    if ok:
        passed += 1

print(f"{passed}/{len(tests)} ward/location tests passed")
