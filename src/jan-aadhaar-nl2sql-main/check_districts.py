import duckdb

conn = duckdb.connect()
df = conn.execute("SELECT DISTRICT_NAME_ENG, COUNT(*) FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' WHERE CASTE IN ('MUSHLIM', 'MUSLIM', '48 FAKIR', '17 DHOBI(MUSLIM)', '41 MOYLA', 'PATHAN') GROUP BY DISTRICT_NAME_ENG").df()
print(df)
