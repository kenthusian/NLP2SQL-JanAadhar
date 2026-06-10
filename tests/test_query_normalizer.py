from normalization.query_normalizer import normalize_query


def test_normalizer_corrects_common_user_typos():
    result = normalize_query("all femail benificiaries in jaipor")
    assert result.normalized == "all female beneficiaries in Jaipur"
    assert result.corrections["femail"] == "female"
    assert result.corrections["benificiaries"] == "beneficiaries"
    assert result.corrections["jaipor"] == "Jaipur"


def test_normalizer_corrects_district_typo():
    result = normalize_query("boys above 21 in jodpur")
    assert "Jodhpur" in result.normalized


def test_normalizer_preserves_valid_natural_language():
    question = "Show all female beneficiaries receiving pension in Jaipur district."
    result = normalize_query(question)
    assert result.normalized == question
    assert result.corrections == {}


def test_normalizer_corrects_location_protected_typos():
    # Location prepositions should allow fuzzy matching districts (using the lower 88 threshold)
    r1 = normalize_query("Show all boys above 21 in Jaipurr")
    assert r1.normalized == "Show all boys above 21 in Jaipur"
    assert r1.corrections["Jaipurr"] == "Jaipur"

    r2 = normalize_query("Show all boys above 21 in Jaipura")
    assert r2.normalized == "Show all boys above 21 in Jaipur"
    assert r2.corrections["Jaipura"] == "Jaipur"

    r3 = normalize_query("from Jodhpura to Ajmerr")
    assert r3.normalized == "from Jodhpur to Ajmer"
    assert r3.corrections["Jodhpura"] == "Jodhpur"
    assert r3.corrections["Ajmerr"] == "Ajmer"


def test_normalizer_preserves_person_protected_names():
    # Person prepositions and quotes should keep the strict 95 threshold and not corrupt names
    # "Alwa" vs "Alwar" scores 88.8, so it should be preserved under person protection context.
    r1 = normalize_query("member named Alwa in Jaipurr")
    assert r1.normalized == "member named Alwa in Jaipur"
    assert r1.corrections == {"Jaipurr": "Jaipur"}

    # Same for quotes (which are also person-protected unless preceded by location preposition)
    r2 = normalize_query("find 'Ajmera' in Jaipurr")
    assert r2.normalized == "find 'Ajmera' in Jaipur"
    assert r2.corrections == {"Jaipurr": "Jaipur"}


