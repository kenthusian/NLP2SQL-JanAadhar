import re
from collections import defaultdict

hints = [
    "OCCUPATION ILIKE '%unemployed%'",
    "(OCCUPATION ILIKE '%state personnel%' OR OCCUPATION ILIKE '%autonomous%')",
    "AGE > 18",
    "AGE < 60",
    "EDUCATION = 'illiterate'",
    "EDUCATION ILIKE '%graduate%'",
    "(ACCOUNT_NO IS NULL OR BANK IS NULL)",
    "CASTE_CATEGORY = 'SC'"
]

def group_hints(hints):
    groups = defaultdict(list)
    for h in hints:
        m = re.match(r"^\s*\(?([A-Z_]+)\b", h)
        if m:
            col = m.group(1)
            groups[col].append(h)
        else:
            groups["OTHER"].append(h)
            
    final_hints = []
    for col, vals in groups.items():
        if col == "OTHER":
            final_hints.extend(vals)
        elif len(vals) == 1:
            final_hints.append(vals[0])
        else:
            # multiple hints for same column
            if col in ["AGE", "INCOME"]:
                # numeric ranges should be ANDed
                joined = " AND ".join(f"({v})" if " OR " in v and not v.startswith("(") else v for v in vals)
                final_hints.append(joined)
            else:
                # categorical traits should be ORed
                joined = " OR ".join(f"({v})" if " AND " in v and not v.startswith("(") else v for v in vals)
                final_hints.append(f"({joined})")
                
    return final_hints

print("Output:")
for h in group_hints(hints):
    print(h)
