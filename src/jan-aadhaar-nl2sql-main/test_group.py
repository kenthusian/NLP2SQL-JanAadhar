import sys
sys.path.append('.')
from llm.fast_sql import _group_domain_hints

hints = [
    "OCCUPATION ILIKE '%unemployed%'",
    "OCCUPATION ILIKE '%state personnel%' OR OCCUPATION ILIKE '%autonomous%' OR OCCUPATION ILIKE '%psu%'",
    "(ACCOUNT_NO IS NULL OR ACCOUNT_NO = '' OR BANK IS NULL OR BANK = '')"
]
print(_group_domain_hints(hints))
