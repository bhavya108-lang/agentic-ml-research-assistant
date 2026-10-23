from app.rag.loader import load_pdf
from app.rag.splitter import split_documents


documents = load_pdf(
    "data/papers/yolo_v1.pdf"
)

chunks = split_documents(documents)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk.page_content)
    print("\nMetadata:")
    print(chunk.metadata)