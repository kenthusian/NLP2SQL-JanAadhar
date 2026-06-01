from prompting.prompt_builder import PromptBuilder
from retrieval.schema_retriever import RetrievalResult


def test_prompt_uses_only_retrieved_columns():
    result = RetrievalResult(
        question="female bank members",
        tables=["member", "bank_details"],
        columns=["member.gender", "member.member_name", "bank_details.bank_name"],
        relationships=[{"from_table": "bank_details", "from_column": "member_id", "to_table": "member", "to_column": "member_id"}],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    assert "member.gender" in prompt
    # Verify 'family' table is NOT listed in the Available tables section.
    # Note: '- family' appears elsewhere in the prompt instructions (e.g. '- family head or HOF'),
    # so we scope the check to the tables section only.
    tables_section = prompt[prompt.index("Available tables:"):prompt.index("Relevant columns:")]
    assert "family" not in tables_section
    assert "Do not invent tables or columns" in prompt


def test_prompt_mentions_business_meaning_for_physical_columns():
    result = RetrievalResult(
        question="boys",
        tables=["member"],
        columns=["member.gender"],
        relationships=[],
        documents=[],
        confidence=0.9,
    )
    prompt = PromptBuilder().build(result)
    assert "business meaning: gender" in prompt
    assert "valid example values: Male, Female" in prompt
