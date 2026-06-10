from validation.sql_validator import SQLValidator


def test_validator_accepts_known_join():
    sql = "SELECT member.member_name FROM member JOIN family ON member.family_id = family.family_id WHERE family.district = 'Jaipur';"
    result = SQLValidator().validate(sql)
    assert result.valid, result.errors


def test_validator_accepts_known_join_with_aliases():
    sql = "SELECT m.member_name FROM member AS m JOIN bank_details AS b ON m.member_id = b.member_id WHERE m.gender = 'Female';"
    result = SQLValidator().validate(sql)
    assert result.valid, result.errors


def test_validator_rejects_hallucinated_column():
    sql = "SELECT member.fake_column FROM member;"
    result = SQLValidator().validate(sql)
    assert not result.valid
    assert "Unknown columns" in "; ".join(result.errors)


def test_validator_rejects_write_statement():
    result = SQLValidator().validate("DROP TABLE member;")
    assert not result.valid


def test_validator_rejects_select_plus_write_statement():
    result = SQLValidator().validate("SELECT member.member_name FROM member; DELETE FROM member;")
    assert not result.valid
    assert "Only one SQL statement is allowed." in result.errors


def test_validator_rejects_select_into():
    result = SQLValidator().validate("SELECT member.member_name INTO backup_member FROM member;")
    assert not result.valid


def test_validator_rejects_qualified_column_from_unjoined_table():
    result = SQLValidator().validate("SELECT member.member_name FROM member WHERE family.district = 'Jaipur';")
    assert not result.valid
    assert "Qualified column references tables not present in FROM/JOIN" in "; ".join(result.errors)


def test_post_process_sql_rewrites_name_equals():
    from app import _post_process_sql
    sql = "SELECT member.member_name FROM member WHERE member.member_name = 'Vijay';"
    processed = _post_process_sql(sql)
    assert "member.member_name LIKE '%Vijay%'" in processed

    # Multi-word name should also be rewritten to LIKE with wildcards
    sql_multi = "SELECT member.member_name FROM member WHERE member.member_name = 'Vijay Kumar Laxmi';"
    processed_multi = _post_process_sql(sql_multi)
    assert "member.member_name LIKE '%Vijay Kumar Laxmi%'" in processed_multi

    # Categorical fields (gender, marital_status, caste_category) should be canonicalized in casing
    sql_cat = "SELECT * FROM member WHERE gender = 'male' AND caste_category = 'obc';"
    processed_cat = _post_process_sql(sql_cat)
    assert "gender = 'Male'" in processed_cat
    assert "caste_category = 'OBC'" in processed_cat

    # District fields should be canonicalized to exact case-insensitive matches from metadata list
    sql_dist = "SELECT * FROM family WHERE district = 'sawai madhopur';"
    processed_dist = _post_process_sql(sql_dist)
    assert "district = 'Sawai Madhopur'" in processed_dist


def test_post_process_sql_prunes_unused_family_join():
    from app import _post_process_sql
    
    # Unused family join should be pruned
    sql = (
        "SELECT DISTINCT member.member_name "
        "FROM member INNER JOIN family ON member.family_id = family.family_id "
        "WHERE member.caste LIKE '%Jat%' AND member.marital_status = 'Widow' AND member.age > 21;"
    )
    processed = _post_process_sql(sql)
    assert "JOIN family" not in processed
    assert "family.family_id" not in processed
    assert "member.family_id = family.family_id" not in processed
    assert "FROM member WHERE (member.caste LIKE" in processed

    # Used family join (by alias) should NOT be pruned
    sql_used_alias = (
        "SELECT DISTINCT member.member_name "
        "FROM member INNER JOIN family F ON member.family_id = F.family_id "
        "WHERE F.district = 'Jaipur';"
    )
    processed_used_alias = _post_process_sql(sql_used_alias)
    assert "JOIN family" in processed_used_alias or "JOIN family F" in processed_used_alias

    # Used family join (no alias) should NOT be pruned
    sql_used_no_alias = (
        "SELECT DISTINCT member.member_name "
        "FROM member INNER JOIN family ON member.family_id = family.family_id "
        "WHERE family.district = 'Jaipur';"
    )
    processed_used_no_alias = _post_process_sql(sql_used_no_alias)
    assert "JOIN family" in processed_used_no_alias


def test_post_process_caste_bilingual_expansion():
    from app import _post_process_sql

    # Test LIKE-based expansion
    sql1 = "SELECT * FROM member WHERE member.caste LIKE '%Rajput%';"
    processed1 = _post_process_sql(sql1)
    assert "member.caste LIKE '%Rajput%'" in processed1
    assert "member.caste LIKE '%Rajpoot%'" in processed1
    assert "member.caste LIKE '%राजपूत%'" in processed1

    # Test equals-based expansion (which gets rewritten to LIKE, then expanded)
    sql2 = "SELECT * FROM member WHERE member.caste = 'Rajput';"
    processed2 = _post_process_sql(sql2)
    assert "member.caste LIKE '%Rajput%'" in processed2
    assert "member.caste LIKE '%Rajpoot%'" in processed2
    assert "member.caste LIKE '%राजपूत%'" in processed2

    # Test IN-based expansion
    sql3 = "SELECT * FROM member WHERE member.caste IN ('Rajput', 'RAJPOOT');"
    processed3 = _post_process_sql(sql3)
    assert "member.caste LIKE '%Rajput%'" in processed3
    assert "member.caste LIKE '%Rajpoot%'" in processed3
    assert "member.caste LIKE '%राजपूत%'" in processed3

    # Test another caste (e.g. Agarwal/Agrawal)
    sql4 = "SELECT * FROM member WHERE member.caste = 'Agarwal';"
    processed4 = _post_process_sql(sql4)
    assert "member.caste LIKE '%Agarwal%'" in processed4
    assert "member.caste LIKE '%Agrawal%'" in processed4
    assert "member.caste LIKE '%अग्रवाल%'" in processed4



