from app.rag.loader import load_pdf


documents = load_pdf("data/papers/yolo_v1.pdf")

print(f"Number of pages loaded: {len(documents)}")

for document in documents[:2]:
    print("\n--- PAGE ---")
    print(document.page_content[:1000])
    print("\nMetadata:")
    print(document.metadata)