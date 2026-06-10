import duckdb

conn = duckdb.connect()
df = conn.execute("SELECT RELATION_WITH_HOF, COUNT(*) as cnt FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' GROUP BY RELATION_WITH_HOF ORDER BY cnt DESC").df()
print(df)
