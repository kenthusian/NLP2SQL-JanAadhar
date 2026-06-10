import duckdb
import json
import os
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, OLLAMA_URL

def extract_distinct_castes() -> list[str]:
    conn = duckdb.connect()
    # Try parquet first
    try:
        parquet_path = str(DATA_DIR / '**' / '*.parquet')
        query = f"SELECT DISTINCT CASTE FROM read_parquet('{parquet_path}') WHERE CASTE IS NOT NULL"
        df = conn.execute(query).df()
    except Exception as e:
        print("Falling back to CSV:", e)
        # fallback to CSV for dummy dataset
        csv_path = Path(__file__).resolve().parents[1] / 'Dummy_Data_Set.csv'
        query = f"SELECT DISTINCT CASTE FROM read_csv_auto('{str(csv_path)}') WHERE CASTE IS NOT NULL"
        df = conn.execute(query).df()
        
    castes = df['CASTE'].tolist()
    # Clean up and strip whitespace
    castes = [c.strip() for c in castes if isinstance(c, str) and c.strip()]
    return castes

def group_castes_with_ollama(castes: list[str]) -> dict[str, list[str]]:
    prompt = f"""You are a data standardization expert. I have a list of raw caste names from a database in Rajasthan, India. They contain misspellings, numeric prefixes, and Hindi text (Devanagari). 
Please cluster these raw caste names into standard, normalized English string keys. 
For example, group 'RAJPUT', 'RAJPOOT', 'राजपूत ', '14 RAJPUT' under the key 'rajput'.
Group 'अग्रवाल', 'AGARWAL', 'AGRAWAL' under the key 'agarwal'.

Return ONLY a valid JSON object where keys are the normalized english string, and values are arrays of the raw strings that belong to it. No markdown formatting.

Raw Castes:
{json.dumps(castes, ensure_ascii=False)}
"""
    payload = {
        "model": "qwen2.5-coder:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2000,
        },
        "format": "json"
    }

    print("Calling Ollama to cluster castes...")
    resp = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120.0)
    resp.raise_for_status()
    
    data = resp.json()
    output_str = data.get("response", "").strip()
    
    try:
        # qwen might return markdown fences even with format=json
        if output_str.startswith("```json"):
            output_str = output_str[7:]
        if output_str.startswith("```"):
            output_str = output_str[3:]
        if output_str.endswith("```"):
            output_str = output_str[:-3]
        
        mapping = json.loads(output_str.strip())
        return mapping
    except json.JSONDecodeError as e:
        print("Failed to decode JSON from LLM. Raw output:", output_str)
        raise e

def main():
    castes = extract_distinct_castes()
    print(f"Extracted {len(castes)} distinct castes.")
    
    mapping = group_castes_with_ollama(castes)
    
    # Verify the mapping coverage
    mapped_count = sum(len(v) for v in mapping.values())
    print(f"Mapped {mapped_count} out of {len(castes)} castes.")
    
    # Save the mapping
    output_path = Path(__file__).resolve().parents[1] / 'data' / 'caste_mapping.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved mapping to {output_path}")

if __name__ == '__main__':
    main()
