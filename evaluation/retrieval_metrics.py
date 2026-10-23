import json
import math
from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.bm25 import create_bm25_retriever
from app.rag.vector_store import load_vector_store
from app.rag.retriever import hybrid_search
from evaluation.inspect_chunks import chunk_id, PDF_PATH

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    rel_set = set(relevant_ids)
    return len(top_k.intersection(rel_set)) / len(rel_set)

def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    rel_set = set(relevant_ids)
    for rank, r_id in enumerate(retrieved_ids[:k], start=1):
        if r_id in rel_set:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    rel_set = set(relevant_ids)
    dcg = 0.0
    for i, r_id in enumerate(retrieved_ids[:k]):
        if r_id in rel_set:
            dcg += 1.0 / math.log2(i + 2)  # rank is i+1, log2(rank+1) = log2(i+2)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(eval_file: str = "evaluation/datasets/retrieval_eval.json"):
    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    bm25 = create_bm25_retriever(chunks, k=10)
    vector_store = load_vector_store()

    metrics = {"recall@5": [], "recall@10": [], "mrr@5": [], "ndcg@5": []}

    for item in dataset:
        results = hybrid_search(
            query=item["query"],
            bm25_retriever=bm25,
            vector_store=vector_store,
            bm25_k=10,
            vector_k=10,
            top_n=10,
        )
        retrieved_ids = [chunk_id(doc) for doc in results]
        rel_ids = item["relevant_chunk_ids"]

        metrics["recall@5"].append(recall_at_k(retrieved_ids, rel_ids, 5))
        metrics["recall@10"].append(recall_at_k(retrieved_ids, rel_ids, 10))
        metrics["mrr@5"].append(mrr_at_k(retrieved_ids, rel_ids, 5))
        metrics["ndcg@5"].append(ndcg_at_k(retrieved_ids, rel_ids, 5))

    return {k: round(sum(v) / len(v), 4) for k, v in metrics.items()}