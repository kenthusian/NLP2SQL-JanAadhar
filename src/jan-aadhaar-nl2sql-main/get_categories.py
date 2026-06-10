import duckdb

conn = duckdb.connect()
df_occ = conn.execute("SELECT OCCUPATION, COUNT(*) as cnt FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' GROUP BY OCCUPATION ORDER BY cnt DESC").df()
df_edu = conn.execute("SELECT EDUCATION, COUNT(*) as cnt FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' GROUP BY EDUCATION ORDER BY cnt DESC").df()

print("=== OCCUPATIONS ===")
print(df_occ.to_string())

print("\n=== EDUCATION ===")
print(df_edu.to_string())
