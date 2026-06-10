import duckdb
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

def extract_villages() -> list[str]:
    conn = duckdb.connect()
    try:
        parquet_path = str(DATA_DIR / '**' / '*.parquet')
        query = f"SELECT DISTINCT VILL_NAME_ENG FROM read_parquet('{parquet_path}') WHERE VILL_NAME_ENG IS NOT NULL AND VILL_NAME_ENG != ''"
        df = conn.execute(query).df()
    except Exception as e:
        print("Falling back to CSV:", e)
        csv_path = Path(__file__).resolve().parents[1] / 'Dummy_Data_Set.csv'
        query = f"SELECT DISTINCT VILL_NAME_ENG FROM read_csv_auto('{str(csv_path)}') WHERE VILL_NAME_ENG IS NOT NULL AND VILL_NAME_ENG != ''"
        df = conn.execute(query).df()
        
    villages = df['VILL_NAME_ENG'].tolist()
    villages = [str(v).strip() for v in villages if str(v).strip()]
    return villages

def main():
    villages = extract_villages()
    print(f"Extracted {len(villages)} distinct villages.")
    
    output_path = Path(__file__).resolve().parents[1] / 'data' / 'villages.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(villages, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved to {output_path}")

if __name__ == '__main__':
    main()
