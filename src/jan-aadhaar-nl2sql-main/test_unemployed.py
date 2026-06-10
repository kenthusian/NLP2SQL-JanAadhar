import httpx

print("Running validation tests...")
queries = [
    "show all unemployed people",
    "show all illiterate people who are unemployed"
]

for q in queries:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
            
    print(f"\nQ: {q}")
    print(f"Source: {r.get('source')}")
    # Write SQL to a file to avoid Windows console encoding issues
    with open("test_unemployed_output.txt", "a", encoding="utf-8") as f:
        f.write(f"\nQ: {q}\n")
        if "LIMIT" in sql:
            f.write(sql.split("WHERE ")[-1].strip() + "\n")
        else:
            f.write(sql + "\n")
    print("SQL written to test_unemployed_output.txt")
