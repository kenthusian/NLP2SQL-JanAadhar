import duckdb
con = duckdb.connect()
print(con.execute("SELECT COUNT(*) AS total_count FROM (SELECT ENROLLMENT_ID FROM read_parquet('data/aadhaar/**/*.parquet') GROUP BY ENROLLMENT_ID HAVING COUNT(*) > 1) sub").df())
