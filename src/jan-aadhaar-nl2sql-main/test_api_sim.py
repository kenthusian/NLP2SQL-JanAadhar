import sys
sys.path.append('.')
from rag.domain_dict import extract_sql_hints
from rag.normalizer import normalize_query
from llm.fast_sql import try_build_sql

query = "show data of all fakirs and muslims in Jaipur"
normalized = normalize_query(query)
domain_hints, mapped_words = extract_sql_hints(normalized)

print("Mapped words:", mapped_words)
sql = try_build_sql(normalized, domain_hints, mapped_words)
print("SQL:", sql)
