from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.bm25 import create_bm25_retriever


documents = load_pdf(
    "data/papers/traffic_sign_detection.pdf"
)

chunks = split_documents(documents)

bm25 = create_bm25_retriever(chunks, k=3)

query = "YOLOv8 TT100K traffic sign detection"

results = bm25.invoke(query)

print(f"Retrieved {len(results)} results.")

for i, result in enumerate(results, start=1):
    print(f"\n========== RESULT {i} ==========")
    print(result.page_content)
    print("\nMetadata:")
    print(result.metadata)