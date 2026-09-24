from pathlib import Path

from langchain_chroma import Chroma

from app.rag.embeddings import get_embeddings


VECTOR_DB_PATH = "data/chroma"


def create_vector_store(chunks):
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name="research_documents",
    )

    return vector_store


def load_vector_store():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="research_documents",
    )