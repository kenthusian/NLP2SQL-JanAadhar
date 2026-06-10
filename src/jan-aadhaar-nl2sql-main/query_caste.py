import duckdb
con = duckdb.connect()
con.execute("CREATE VIEW aadhaar AS SELECT * FROM 'data/aadhaar/**/*.parquet'")
res = con.execute("SELECT DISTINCT CASTE FROM aadhaar WHERE CASTE ILIKE '%saini%' OR CASTE ILIKE '%mali%' LIMIT 10").fetchall()
print(res)
