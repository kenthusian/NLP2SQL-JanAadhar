import duckdb
conn = duckdb.connect()
query = "SELECT CASTE, COUNT(*) as cnt FROM read_csv_auto('Dummy_Data_Set.csv') GROUP BY CASTE ORDER BY cnt DESC LIMIT 50"
df = conn.execute(query).df()
df.to_csv('tmp_castes.csv', index=False, encoding='utf-8')
