import httpx

queries = [
    "show data of Malwani",
]

print("Running validation tests...")
for q in queries:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
            
    print(f"\nQ: {q}")
    if "LIMIT" in sql:
        print(f"SQL: {sql.split('WHERE ')[-1].strip()}")
    else:
        print(f"SQL: {sql}")
