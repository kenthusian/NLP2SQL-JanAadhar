import httpx

print("Running validation tests...")
queries = [
    "show data of all jats",
    "show data of all jats from Pichkarai"
]

for q in queries:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
            
    print(f"\nQ: {q}")
    print(f"Source: {r.get('source')}")
    if "LIMIT" in sql:
        print(f"SQL: {sql.split('WHERE ')[-1].strip()}")
    else:
        print(f"SQL: {sql}")
