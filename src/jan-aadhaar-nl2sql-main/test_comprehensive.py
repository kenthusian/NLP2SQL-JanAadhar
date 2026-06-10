import httpx
import json

queries = [
    "list all unbanked state personnel who are post graduates",
    "count the number of contractual employees with income between 10k and 50k",
    "show all widowed women in Jaipur who are self-employed",
    "list all male students without a bank account",
    "show members with a bank account in SBI",
    "find divorced men from OBC category",
    "find farmers with income above 1 lakh"
]

print("Running validation tests...")

with open("test_comprehensive_output.txt", "w", encoding="utf-8") as f:
    for q in queries:
        print(f"Testing: {q}")
        try:
            r = httpx.post("http://127.0.0.1:8000/query", json={"question": q}, timeout=120).json()
            sql = r.get("sql", "NO SQL")
            f.write(f"\nQ: {q}\n")
            if "LIMIT" in sql:
                f.write(sql.split("WHERE ")[-1].strip() + "\n")
            else:
                f.write(sql + "\n")
        except Exception as e:
            f.write(f"\nQ: {q}\nERROR: {e}\n")

print("Done. Output written to test_comprehensive_output.txt")
