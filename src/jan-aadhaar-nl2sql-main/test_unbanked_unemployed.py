import httpx

q = "count unbanked unemployed state personnel"
print("Running validation tests...")

try:
    r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
    sql = r.get("sql", "NO SQL")
    
    with open("test_unbanked_unemployed_output.txt", "w", encoding="utf-8") as f:
        f.write(f"\nQ: {q}\n")
        f.write(sql + "\n")
except Exception as e:
    print(f"Error: {e}")

print("Done. Output written to test_unbanked_unemployed_output.txt")
