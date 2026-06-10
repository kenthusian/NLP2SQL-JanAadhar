import httpx

queries = [
    "show all beople from 2rpm",
    "how many famlies have 1 or more than 1 children",
    "shw me all ricch peeple in jaipor",
    "find all unmarred womens above 50",
    "gimme total count of famers with icome > 1lak",
    "details of people named rahul in rural area",
    "who is the oldest person in sbi bank",
    "smallest famliy",
    "highest earning st caste member",
    "how many graduate females in ajmr",
    "give me data of all illiterat laborrs",
    "list all widows with 0 incom",
    "show youth between 18 and 35 in udiapur",
    "find all m.a degree holders",
    "how many people don't have a bank account",
    "shw me minrs with pnb bank"
]

def main():
    results = []
    with httpx.Client(timeout=60.0) as client:
        for q in queries:
            print(f"Testing: {q}")
            try:
                res = client.post("http://127.0.0.1:8000/query", json={"question": q})
                if res.status_code == 200:
                    data = res.json()
                    sql = data.get("sql", "").replace("\n", " ")
                    results.append(f"Q: {q}\n   SQL: {sql}\n   Rows: {data.get('row_count')}\n   Source: {data.get('source')}\n")
                else:
                    results.append(f"Q: {q}\n   ERROR: HTTP {res.status_code}\n")
            except Exception as e:
                results.append(f"Q: {q}\n   ERROR: {str(e)}\n")

    with open("weird_queries.txt", "w", encoding="utf-8") as f:
        f.write("=== Weird Queries Test Results ===\n\n")
        f.write("\n".join(results))
    print("Done! Results saved to weird_queries.txt")

if __name__ == "__main__":
    main()
