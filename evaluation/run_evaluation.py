# evaluation/run_evaluation.py
import json
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.tool_metrics import evaluate_tools
from evaluation.latency_metrics import measure_workflow_latency
from evaluation.citation_metrics import evaluate_citations
from evaluation.answer_rubric import evaluate_answer_usefulness

def main():
    test_query = "Explain how the YOLO architecture predicts bounding boxes and discuss its main limitations compared to Fast R-CNN."

    print("1. Evaluating Hybrid Retrieval (Recall@5, Recall@10, MRR, nDCG)...")
    retrieval_scores = evaluate_retrieval()

    print("2. Evaluating External Tool Reliability & Tool Latency...")
    tool_queries = [
        "YOLO object detection architecture",
        "PASCAL VOC 2007 dataset",
        test_query,  # Tests how tools handle a full conversational sentence
    ]
    tool_scores = evaluate_tools(tool_queries, max_results=2)

    print("3. Running End-to-End Workflow & Profiling Latency...")
    latency_data = measure_workflow_latency(test_query)
    final_state = latency_data.pop("final_state")

    print("4. Evaluating Citation Fidelity...")
    citation_scores = evaluate_citations(
        final_report=final_state["final_report"],
        local_evidence=final_state["local_evidence"],
        research_sources=final_state["research_sources"],
    )

    print("5. Grading Answer Usefulness (1-5 Rubric)...")
    usefulness_scores = evaluate_answer_usefulness(
        query=test_query,
        final_report=final_state["final_report"],
        local_evidence=final_state["local_evidence"],
        research_sources=final_state["research_sources"],
    )

    summary = {
        "retrieval_quality": retrieval_scores,
        "tool_reliability": tool_scores,
        "latency": latency_data,
        "citation_fidelity": citation_scores,
        "answer_usefulness": usefulness_scores,
    }

    print("\n================ EVALUATION SUMMARY ================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()