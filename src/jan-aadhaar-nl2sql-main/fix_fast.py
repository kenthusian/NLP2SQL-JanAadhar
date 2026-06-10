def fix_fast_sql():
    with open('llm/fast_sql.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    import re
    # We want to replace the broken block inside 7.5
    # Since I don't want to use regex across multiple lines, I will just do string replacement
    
    # Let's extract the first half up to "if _is_list(q) and not _groupby_col(q) and has_unmapped:"
    part1 = content.split("    # ── 7.5 Generic Name/Caste search (FAST PATH) ─────────────────────────────\n    if _is_list(q) and not _groupby_col(q) and has_unmapped:\n")[0]
    
    # And we know where 8. Pure group-by starts
    part2 = content.split("    # ── 8. Pure group-by (no domain hints needed) ─────────────────────────────\n")[1]
    
    new_7_5 = """    # ── 7.5 Generic Name/Caste search (FAST PATH) ─────────────────────────────
    if _is_list(q) and not _groupby_col(q) and has_unmapped:
        # Check if the unmapped words might just be a generic noun search
        complex_words = {"not", "without", "except", "but", "only", "highest", "lowest", "top", "bottom"}
        q_words = set(re.sub(r"[^\w\-\s]", "", q).lower().split())
        
        if not (complex_words & q_words):
            unmapped_conditions = []
            q_lower = re.sub(r"[^\w\-\s]", "", q.lower()) # stripped version for regex
            
            for word in unmapped_words:
                term_escaped = word.replace("'", "''")
                is_loc = bool(re.search(rf"\\b(from|in|village|city|district|place|lives in)\\s+{re.escape(word)}\\b", q_lower))
                is_name = bool(re.search(rf"\\b(named|called|person|name|who is)\\s+{re.escape(word)}\\b", q_lower))
                is_caste = bool(re.search(rf"\\b(caste|surname|community)\\s+{re.escape(word)}\\b", q_lower))
                
                if is_loc:
                    name_hint = (
                        f"(VILL_NAME_ENG ILIKE '%{term_escaped}%' OR GP_NAME_ENG ILIKE '%{term_escaped}%' "
                        f"OR BLOCK_NAME_ENG ILIKE '%{term_escaped}%' OR DISTRICT_NAME_ENG ILIKE '%{term_escaped}%')"
                    )
                elif is_name or is_caste:
                    name_hint = f"(NAME_EN ILIKE '%{term_escaped}%' OR CASTE ILIKE '%{term_escaped}%')"
                else:
                    name_hint = (
                        f"(CASTE ILIKE '%{term_escaped}%' OR NAME_EN ILIKE '%{term_escaped}%' "
                        f"OR VILL_NAME_ENG ILIKE '%{term_escaped}%' OR GP_NAME_ENG ILIKE '%{term_escaped}%' "
                        f"OR BLOCK_NAME_ENG ILIKE '%{term_escaped}%' OR DISTRICT_NAME_ENG ILIKE '%{term_escaped}%')"
                    )
                unmapped_conditions.append(name_hint)
            
            combined_unmapped = " AND ".join(f"({c})" for c in unmapped_conditions)
            final_where = f"{where_clause} AND ({combined_unmapped})" if where_clause else f"WHERE {combined_unmapped}"
            
            return (
                f"SELECT {_DEFAULT_COLS}\\n"
                f"FROM {TABLE_NAME}\\n"
                f"{final_where}\\n"
                f"LIMIT 500;"
            )

"""
    
    new_content = part1 + new_7_5 + "    # ── 8. Pure group-by (no domain hints needed) ─────────────────────────────\n" + part2
    
    with open('llm/fast_sql.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
if __name__ == '__main__':
    fix_fast_sql()
