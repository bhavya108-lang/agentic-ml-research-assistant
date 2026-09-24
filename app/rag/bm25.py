from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever(
    chunks: list[Document],
    k: int = 4,
):
    retriever = BM25Retriever.from_documents(chunks)

    retriever.k = k

    return retriever