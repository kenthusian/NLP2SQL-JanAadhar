import sys, duckdb
sys.stdout.reconfigure(encoding="utf-8")
con = duckdb.connect(":memory:")
con.execute(
    "CREATE VIEW aadhaar AS "
    "SELECT * FROM read_parquet('data/aadhaar/**/*.parquet', hive_partitioning=true)"
)
print("Total rows:", con.execute("SELECT COUNT(*) FROM aadhaar").fetchone()[0])
print("Districts:", con.execute("SELECT COUNT(DISTINCT DISTRICT_NAME_ENG) FROM aadhaar").fetchone()[0])
cols = [d[0] for d in con.execute("DESCRIBE aadhaar").fetchall()]
print("Columns:", cols)
