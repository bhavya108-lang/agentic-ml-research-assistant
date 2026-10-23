import json
import re
from app.llm.provider import get_llm
from app.graph.workflow import get_response_text

def evaluate_answer_usefulness(
    query: str, final_report: str, local_evidence: str, research_sources: str
) -> dict:
    judge_llm = get_llm(max_tokens=1000)
    prompt = f"""You are an expert ML Research Evaluator. Grade the FINAL REPORT on a 1 to 5 integer scale for each rubric dimension.

QUERY: {query}

EVIDENCE CONTEXT:
{local_evidence}
{research_sources}

FINAL REPORT:
{final_report}

RUBRIC (1 = Poor, 3 = Acceptable, 5 = Excellent):
- correctness (1-5): Technical accuracy of ML/CV concepts, architectures, and metrics.
- completeness (1-5): Covers all 6 required report sections, datasets, models, training, and trade-offs.
- reasoning_quality (1-5): Synthesizes local and external evidence logically; justifies experimental choices.
- factualness (1-5): Adheres strictly to provided evidence without inventing SOTA claims or numbers.

Return ONLY valid JSON:
{{
  "correctness": <int 1-5>,
  "completeness": <int 1-5>,
  "reasoning_quality": <int 1-5>,
  "factualness": <int 1-5>,
  "overall_score": <float average>,
  "justification": "<1-2 sentence explanation>"
}}"""

    response = get_response_text(judge_llm.invoke(prompt)).strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"error": "Failed to parse judge output", "raw": response}
    return json.loads(match.group(0))