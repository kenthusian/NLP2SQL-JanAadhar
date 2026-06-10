import sqlglot
import sqlglot.expressions as exp

sql1 = "SELECT * FROM citizen WHERE district = 'Srinagar' AND district = 'Jaipur'"
sql2 = "SELECT * FROM citizen WHERE gender = 'Male' AND gender = 'Female'"
sql3 = "SELECT * FROM citizen WHERE district LIKE '%A%' AND district LIKE '%B%' AND age > 50"

def fix_mutually_exclusive_and(tree):
    # This is a bit tricky because AND is binary.
    # sqlglot has a function to flatten ANDs.
    pass

    def transformer(node):
        if isinstance(node, exp.Where):
            # Flatten the AND tree into a list of conditions
            if isinstance(node.this, exp.And):
                conds = list(node.this.flatten())
            else:
                conds = [node.this]
            
            # Group conditions by column name
            col_conds = {}
            other_conds = []
            
            for c in conds:
                # We only care about EQ and LIKE on a direct column
                if isinstance(c, (exp.EQ, exp.Like)):
                    if isinstance(c.left, exp.Column):
                        col_name = c.left.name.lower()
                        if col_name not in col_conds:
                            col_conds[col_name] = []
                        col_conds[col_name].append(c)
                        continue
                other_conds.append(c)
            
            # Rebuild the conditions
            new_conds = []
            for col_name, c_list in col_conds.items():
                if len(c_list) > 1:
                    # Convert AND to OR for these mutually exclusive conditions
                    new_conds.append(exp.Paren(this=exp.or_(*c_list)))
                else:
                    new_conds.append(c_list[0])
            
            new_conds.extend(other_conds)
            
            if new_conds:
                node.set('this', exp.and_(*new_conds))
        return node
    
    return tree.transform(transformer)

for sql in [sql1, sql2, sql3]:
    tree = sqlglot.parse_one(sql, dialect='sqlite')
    print("Input: ", sql)
    print("Output:", fix_mutually_exclusive_and(tree).sql(dialect='sqlite'))
    print()
