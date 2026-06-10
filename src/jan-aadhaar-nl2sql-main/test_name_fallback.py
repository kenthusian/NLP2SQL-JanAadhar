import httpx
import time

queries = [
    "show data of Mamaraj",
    "show data of Mamaraj from Jaipur",
]

print("Running validation tests...")
for q in queries:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
    
    steps = r.get("pipeline_steps", [])
    cache_status = "MISS"
    for s in steps:
        if "Deterministic" in s["name"] and s["status"] == "used":
            print(f"FAST PATH USED")
            
    print(f"\nQ: {q}")
    if "LIMIT" in sql:
        print(f"SQL: {sql.split('WHERE ')[-1].strip()}")
    else:
        print(f"SQL: {sql}")
    time.sleep(0.5)
