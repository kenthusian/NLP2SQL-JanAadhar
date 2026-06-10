import asyncio
from llm.prompt_builder import build_prompt
from llm.ollama_client import generate_sql

async def main():
    prompt = build_prompt(
        "show list of all households with 1 or more than 1 children", 
        [], 
        ["RELATION_WITH_HOF ILIKE '%Son%' OR RELATION_WITH_HOF ILIKE '%Daughter%'"]
    )
    sql, _ = await generate_sql(prompt)
    print("LLM Output:", sql)

asyncio.run(main())
