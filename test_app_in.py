from app import _post_process_sql

queries = [
    "SELECT * FROM citizen WHERE district IN ('Srinagar', 'Beejasar')",
    "SELECT * FROM citizen WHERE district IN ('Nagaur', 'Jayal')"
]

for q in queries:
    print(f'Input:  {q}')
    print(f'Output: {_post_process_sql(q)}')
    print()
