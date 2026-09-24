from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.bm25 import create_bm25_retriever
from app.rag.vector_store import load_vector_store
from app.rag.retriever import hybrid_search


PDF_PATH = "data/papers/traffic_sign_detection.pdf"


def retrieve_evidence(
    query: str,
    top_n: int = 5,
):
    """
    Retrieve evidence using hybrid BM25 + vector search.
    """

    # Load and split research documents
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)

    # Create BM25 retriever
    bm25 = create_bm25_retriever(
        chunks,
        k=5,
    )

    # Load Chroma vector store
    vector_store = load_vector_store()

    # Hybrid retrieval
    retrieved_documents = hybrid_search(
        query=query,
        bm25_retriever=bm25,
        vector_store=vector_store,
        bm25_k=5,
        vector_k=5,
        top_n=top_n,
    )

    # Format evidence with source metadata
    evidence_parts = []

    for i, doc in enumerate(retrieved_documents, start=1):
        source = doc.metadata.get(
            "source",
            "Unknown source",
        )

        page = doc.metadata.get(
            "page",
            "Unknown page",
        )

        evidence_parts.append(
            f"""--- Evidence {i} ---
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(evidence_parts)