import httpx
r = httpx.post("http://127.0.0.1:8000/query", json={"question": "show data of all jats from Jaipur and Bikaner"}, timeout=120).json()
print("SQL:", r.get("sql", "NO SQL"))
print("Steps:", r.get("pipeline_steps", []))
