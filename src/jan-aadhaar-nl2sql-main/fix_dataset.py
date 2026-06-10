import os
import glob
import pandas as pd
import duckdb

parquet_files = glob.glob('data/aadhaar/**/*.parquet', recursive=True)

for file in parquet_files:
    df = pd.read_parquet(file)
    # The actual Family ID is embedded at the end of the ENROLLMENT_ID, e.g., '0005-1563-13488' -> '13488'
    df['ENROLLMENT_ID'] = df['ENROLLMENT_ID'].apply(lambda x: x.split('-')[-1] if isinstance(x, str) and '-' in x else x)
    df.to_parquet(file, index=False)
    print(f"Updated {file}")

print("All parquet files updated successfully!")
