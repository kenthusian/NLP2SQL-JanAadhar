import duckdb

conn = duckdb.connect()
df = conn.execute("SELECT CASTE, COUNT(*) as cnt FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' WHERE CASTE IS NOT NULL GROUP BY CASTE ORDER BY cnt DESC").df()

print(f"Total unique castes: {len(df)}")
df.to_csv("castes_list.csv", index=False)
