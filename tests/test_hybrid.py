from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.bm25 import create_bm25_retriever
from app.rag.vector_store import load_vector_store
from app.rag.retriever import hybrid_search


documents = load_pdf(
    "data/papers/yolo_v1.pdf"
)

chunks = split_documents(documents)

bm25 = create_bm25_retriever(
    chunks,
    k=5,
)

vector_store = load_vector_store()

query = "How does traffic sign detection handle difficult weather and lighting conditions?"

results = hybrid_search(
    query=query,
    bm25_retriever=bm25,
    vector_store=vector_store,
    bm25_k=5,
    vector_k=5,
    top_n=5,
)

print(f"Retrieved {len(results)} hybrid results.")

for i, result in enumerate(results, start=1):

    print(f"\n========== RESULT {i} ==========")

    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)