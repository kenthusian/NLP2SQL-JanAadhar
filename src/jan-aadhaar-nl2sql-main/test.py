import re

def _group(hints):
    from collections import defaultdict
    groups = defaultdict(list)
    others = []
    for h in hints:
        m = re.match(r"^([A-Z_]+)\s*=\s*('.+?'|\d+)$", h.strip())
        if m:
            groups[m.group(1)].append(m.group(2))
        else:
            others.append(h)
    
    final = others.copy()
    for col, vals in groups.items():
        if len(vals) > 1:
            final.append(f"{col} IN ({', '.join(vals)})")
        else:
            final.append(f"{col} = {vals[0]}")
    return final

print(_group(['AGE BETWEEN 18 AND 60', "DISTRICT_NAME_ENG = 'Bikaner'", "DISTRICT_NAME_ENG = 'Jaipur'", "DISTRICT_NAME_ENG = 'Jodhpur'"]))
