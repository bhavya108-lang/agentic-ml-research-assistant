from app.rag.vector_store import load_vector_store


vector_store = load_vector_store()

query = "How does the model handle difficult weather and lighting conditions?"

results = vector_store.similarity_search(
    query,
    k=3,
)

print(f"Retrieved {len(results)} results.")

for i, result in enumerate(results, start=1):
    print(f"\n========== RESULT {i} ==========")
    print(result.page_content)
    print("\nMetadata:")
    print(result.metadata)