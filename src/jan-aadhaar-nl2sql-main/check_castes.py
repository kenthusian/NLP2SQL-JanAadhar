import json
import pandas as pd
import re

with open("data/caste_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

# flatten all mapped values
all_mapped_values = set()
for values in mapping.values():
    for v in values:
        all_mapped_values.add(v.strip())

df = pd.read_csv("castes_list.csv")
missing = []
for caste in df["CASTE"]:
    if pd.isna(caste): continue
    caste_str = str(caste).strip()
    if caste_str not in all_mapped_values:
        missing.append(caste_str)

print("Missing from mapping:")
for m in missing:
    print(m)

# Let's also check if they are mapped by canonical name or something.
