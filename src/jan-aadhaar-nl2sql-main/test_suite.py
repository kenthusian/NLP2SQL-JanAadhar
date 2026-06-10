import httpx
import time

queries = [
    "show data of all jats",
    "show data of all jats",  # Should hit cache
    "show data of all jats from Jaipur",  # Should miss cache, have Jaipur
    "show data of all jats from Jaipur and Bikaner",  # Should miss cache, have both
    "show data of all sharmas",  # Should only have Sharma
    "show data of all sharmas from Jaipur",  # Sharma + Jaipur
    "show data of all jaats",  # Should hit cache for 'jats' since domain hints match
]

print("Running validation tests...")
for q in queries:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
    
    # check cache status
    steps = r.get("pipeline_steps", [])
    cache_status = "MISS"
    for s in steps:
        if "Cache" in s["name"] and s["status"] == "hit":
            cache_status = "HIT (" + s["name"] + ")"
            
    print(f"\nQ: {q}")
    print(f"Cache: {cache_status}")
    if "LIMIT" in sql:
        print(f"SQL: {sql.split('WHERE ')[-1].strip()}")
    else:
        print(f"SQL: {sql}")
    time.sleep(0.5)
