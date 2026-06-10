import sqlglot
import sqlglot.expressions as exp
from typing import Optional
from nlp.entity_extractor import EntityExtractor, ExtractionResult, Condition

class FastPathEngine:
    def __init__(self):
        self.extractor = EntityExtractor()

    def generate_sql_fast(self, question: str) -> Optional[str]:
        """Option 3: Generates SQL instantly if the query is simple and fully understood."""
        res = self.extractor.extract(question)
        
        # Abort if query contains complex reasoning keywords or unhandled nouns
        if res.has_complex_keywords or len(res.unhandled_words) > 0:
            return None
            
        if not res.entities:
            return None
            
        # Abort if > 3 entity filters to remain safe and defer to LLM for complex joins/logic
        if len(res.entities) > 3:
            return None

        conditions = []
        for col, conds in res.entities.items():
            for cond in conds:
                if cond.operator == "=":
                    if isinstance(cond.value, int):
                        conditions.append(exp.EQ(this=exp.column(col), expression=exp.Literal.number(cond.value)))
                    else:
                        conditions.append(exp.EQ(this=exp.column(col), expression=exp.Literal.string(cond.value)))
                elif cond.operator in (">", "<", ">=", "<="):
                    # Using sqlglot builder methods
                    col_exp = exp.column(col)
                    val_exp = exp.Literal.number(cond.value)
                    if cond.operator == ">": conditions.append(exp.GT(this=col_exp, expression=val_exp))
                    elif cond.operator == "<": conditions.append(exp.LT(this=col_exp, expression=val_exp))
                    elif cond.operator == ">=": conditions.append(exp.GTE(this=col_exp, expression=val_exp))
                    elif cond.operator == "<=": conditions.append(exp.LTE(this=col_exp, expression=val_exp))
                elif cond.operator == "BETWEEN":
                    conditions.append(exp.Between(this=exp.column(col), low=exp.Literal.number(cond.value[0]), high=exp.Literal.number(cond.value[1])))
                elif cond.operator == "LIKE":
                    conditions.append(exp.Like(this=exp.column(col), expression=exp.Literal.string(f"%{cond.value}%")))
                elif cond.operator == "IS NULL":
                    conditions.append(exp.Is(this=exp.column(col), expression=exp.null()))
                elif cond.operator == "IS NOT NULL":
                    conditions.append(exp.Is(this=exp.column(col), expression=exp.Not(this=exp.null())))
                elif cond.operator == "IN":
                    val_list = [exp.Literal.string(v) if isinstance(v, str) else exp.Literal.number(v) for v in cond.value]
                    conditions.append(exp.In(this=exp.column(col), expressions=val_list))
                elif cond.operator == "LIKE_ANY":
                    likes = [exp.Like(this=exp.column(col), expression=exp.Literal.string(f"%{v}%")) for v in cond.value]
                    conditions.append(exp.Or(this=likes[0], expression=likes[1]) if len(likes)==2 else exp.or_(*likes))

        if not conditions:
            return None

        # Build query
        where_clause = exp.and_(*conditions) if len(conditions) > 1 else conditions[0]
        select_cols = [exp.column("member_name"), exp.column("age"), exp.column("gender"), exp.column("district")]
        query = exp.select(*select_cols).from_("citizen").where(where_clause)
        
        return query.sql(dialect="sqlite") + ";"

    def swap_ast_parameters(self, cached_sql: str, cached_question: str, new_question: str) -> Optional[str]:
        """Option 2: Smart Cache parameter swapping for structurally identical queries."""
        old_res = self.extractor.extract(cached_question)
        new_res = self.extractor.extract(new_question)
        
        delta = {}
        # Find which columns had their single value changed
        for col, new_conds in new_res.entities.items():
            old_conds = old_res.entities.get(col, [])
            # Only support swapping single "=" or "LIKE" condition literals for safety
            if len(new_conds) == 1 and len(old_conds) == 1:
                if new_conds[0].operator == old_conds[0].operator and new_conds[0].operator in ("=", "LIKE"):
                    if new_conds[0].value != old_conds[0].value:
                        delta[col] = new_conds[0].value

        if not delta:
            return None # Nothing to swap securely

        # AST Surgery
        try:
            tree = sqlglot.parse_one(cached_sql, dialect="sqlite")
        except:
            return None
            
        swap_count = 0
        
        def transformer(node):
            nonlocal swap_count
            if isinstance(node, (exp.EQ, exp.Like)):
                left = node.left
                right = node.right
                if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                    col_name = left.name.lower()
                    if col_name in delta:
                        new_val = delta[col_name]
                        if isinstance(new_val, int):
                            right.set("this", str(new_val))
                        else:
                            # Keep % if it's a LIKE
                            if "%" in right.name:
                                right.set("this", f"%{new_val}%")
                            else:
                                right.set("this", new_val)
                        swap_count += 1
            return node

        tree = tree.transform(transformer)
        
        # We only succeed if we actually swapped everything we intended to
        if swap_count > 0:
            return tree.sql(dialect="sqlite") + ";"
        return None
