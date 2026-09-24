from langchain_core.documents import Document


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:

    scores = {}
    documents = {}

    for results in result_lists:

        for rank, document in enumerate(results, start=1):

            # Use page content as the document identity.
            doc_id = document.page_content

            if doc_id not in scores:
                scores[doc_id] = 0
                documents[doc_id] = document

            scores[doc_id] += 1 / (k + rank)

    ranked_documents = sorted(
        documents.values(),
        key=lambda document: scores[document.page_content],
        reverse=True,
    )

    return ranked_documents[:top_n]


def hybrid_search(
    query: str,
    bm25_retriever,
    vector_store,
    bm25_k: int = 5,
    vector_k: int = 5,
    top_n: int = 5,
):

    bm25_results = bm25_retriever.invoke(query)

    vector_results = vector_store.similarity_search(
        query,
        k=vector_k,
    )

    return reciprocal_rank_fusion(
        [
            bm25_results[:bm25_k],
            vector_results[:vector_k],
        ],
        top_n=top_n,
    )