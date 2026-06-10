import duckdb

conn = duckdb.connect()
df = conn.execute("SELECT MINORITY, CASTE, DISTRICT_NAME_ENG FROM 'c:/Users/Arav Kilak/OneDrive/Documents/Projects/jan-aadhaar-nl2sql/data/aadhaar/**/*.parquet' WHERE CASTE IN ('MUSHLIM', 'MUSLIM', '48 FAKIR', '17 DHOBI(MUSLIM)', '41 MOYLA', 'PATHAN') AND DISTRICT_NAME_ENG = 'Jaipur'").df()
print(df)
