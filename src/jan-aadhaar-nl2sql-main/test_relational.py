import httpx

QUERIES = [
    "sons of people who have a bank in sbi and are JAT",
    "daughters of farmers who live in Jaipur",
    "wives of men who are state personnel",
    "families where the head is illiterate"
]

for q in QUERIES:
    res = httpx.post('http://127.0.0.1:8000/query', json={'question': q}, timeout=600).json()
    print(f"\nQ: {q}")
    print(f"SQL:\n{res.get('sql')}")
