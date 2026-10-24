from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
)


def _doc_key(doc: Document):
    m = doc.metadata
    if m.get("source") is not None and m.get("start_index") is not None:
        return (m["source"], m["start_index"])
    if m.get("source") is not None and m.get("page") is not None:
        return (m["source"], m["page"], doc.page_content[:200])
    return doc.page_content


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 5,
) -> list[tuple[Document, float]]:
    scores: dict = {}
    documents: dict = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [(documents[key], scores[key]) for key in ranked]


def hybrid_search(
    query: str,
    bm25_retriever,
    vector_store,
    bm25_k: int = 10,
    vector_k: int = 10,
    top_n: int = 5,
) -> list[Document]:
    bm25_results = bm25_retriever.invoke(query)[:bm25_k]
    vector_results = vector_store.similarity_search(query, k=vector_k)

    pool = reciprocal_rank_fusion(
        [bm25_results, vector_results],
        top_n=bm25_k + vector_k,
    )
    if not pool:
        return []

    pairs = [[query, doc.page_content] for doc, _ in pool]
    rerank_scores = cross_encoder.predict(pairs, batch_size=16)

    reranked = [
        Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "rrf_score": rrf,
                "rerank_score": float(score),
            },
        )
        for (doc, rrf), score in zip(pool, rerank_scores)
    ]
    reranked.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return reranked[:top_n]