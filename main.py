from app.llm.provider import get_llm


llm = get_llm("gemini")

response = llm.invoke(
    "Explain RAG in the context of a computer vision research assistant in 3 sentences."
)

print(response.content)