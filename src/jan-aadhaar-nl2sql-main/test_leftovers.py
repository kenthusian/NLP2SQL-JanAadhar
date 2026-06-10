import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.domain_dict import extract_sql_hints
from llm.fast_sql import try_build_sql

queries = [
    "show all rich people in Jaipur",
    "details of people named rahul in rural area",
    "show all people from 2rpm",
    "find all unmarried women above 50",
    "show people NOT from jaipur"
]

for q in queries:
    print(f"\n--- {q} ---")
    hints, mapped_words = extract_sql_hints(q)
    print("Hints:", hints)
    print("Mapped Words:", mapped_words)
    sql = try_build_sql(q, hints, mapped_words)
    if sql:
        print("SQL [DETERMINISTIC]:", sql.replace('\n', ' '))
    else:
        print("SQL [FALLBACK TO LLM]")
