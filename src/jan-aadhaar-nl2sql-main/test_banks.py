import duckdb
con = duckdb.connect()
print(con.execute("SELECT COUNT(*) FROM read_parquet('data/aadhaar/**/*.parquet') WHERE BANK ILIKE '%State Bank of India%' OR IFSC_CODE ILIKE 'SBIN%'").df())
