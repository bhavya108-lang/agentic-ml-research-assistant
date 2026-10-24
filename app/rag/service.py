from functools import lru_cache
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.bm25 import create_bm25_retriever
from app.rag.vector_store import load_vector_store
from app.rag.retriever import hybrid_search

PDF_PATH = "data/papers/yolo_v1.pdf"

@lru_cache(maxsize=1)
def get_bm25_retriever():
    """Caches the BM25 index in memory so we don't re-parse the PDF on every query."""
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    # The 'k' here is a default; we override it dynamically in hybrid_search
    return create_bm25_retriever(chunks, k=5)

@lru_cache(maxsize=1)
def get_chroma_store():
    """Caches the Chroma DB connection."""
    return load_vector_store()

def retrieve_evidence(
    query: str,
    top_n: int = 5,
) -> str:
    """
    Retrieve evidence using hybrid BM25 + vector search.
    """
    bm25 = get_bm25_retriever()
    vector_store = get_chroma_store()

    # Fix for Total Corpus Size Trap: ensure we pull enough candidates from 
    # both sparse and dense retrievers before fusing them, even if top_n is large.
    search_k = max(15, top_n * 3)

    # Hybrid retrieval
    retrieved_documents = hybrid_search(
        query=query,
        bm25_retriever=bm25,
        vector_store=vector_store,
        bm25_k=search_k,
        vector_k=search_k,
        top_n=top_n,
    )

    # Format evidence with source metadata
    evidence_parts = []
    for i, doc in enumerate(retrieved_documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        evidence_parts.append(
            f"--- Evidence {i} ---\nSource: {source} Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(evidence_parts)