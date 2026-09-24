from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vector_store import create_vector_store


documents = load_pdf(
    "data/papers/traffic_sign_detection.pdf"
)

chunks = split_documents(documents)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

vector_store = create_vector_store(chunks)

print("Vector store created successfully.")