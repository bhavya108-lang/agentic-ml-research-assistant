import json
import re
from app.llm.provider import get_llm
from app.graph.workflow import get_response_text

def evaluate_citations(final_report: str, local_evidence: str, research_sources: str) -> dict:
    # 1. Deterministic check: Did the LLM invent citation numbers that don't exist?
    cited_ev_nums = {int(x) for x in re.findall(r"\[Evidence (\d+)\]", final_report)}
    valid_ev_nums = {int(x) for x in re.findall(r"--- Evidence (\d+) ---", local_evidence)}
    
    cited_src_nums = {int(x) for x in re.findall(r"\[Research Source (\d+)\]", final_report)}
    valid_src_nums = {int(x) for x in re.findall(r"--- Research Source (\d+) ---", research_sources)}

    phantom_citations = len(cited_ev_nums - valid_ev_nums) + len(cited_src_nums - valid_src_nums)

    # 2. LLM-as-a-Judge for claim-level grounding
    judge_llm = get_llm(max_tokens=1000)
    prompt = f"""You are an strict scientific fact-checker.
Evaluate the factual claims in the FINAL REPORT against the PROVIDED EVIDENCE.
Ignore section headers and pure experimental recommendations labeled as reasoning.

PROVIDED LOCAL EVIDENCE:
{local_evidence}

PROVIDED EXTERNAL SOURCES:
{research_sources}

FINAL REPORT:
{final_report}

Return ONLY valid JSON with this exact structure:
{{
  "total_claims": <int>,
  "grounded_claims": <int>,
  "unsupported_claim_count": <int>,
  "unsupported_examples": ["<claim 1>", "<claim 2>"]
}}"""
    
    response = get_response_text(judge_llm.invoke(prompt)).strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    data = json.loads(match.group(0)) if match else {
        "total_claims": 0, "grounded_claims": 0, "unsupported_claim_count": 0, "unsupported_examples": []
    }
    
    total = max(data.get("total_claims", 0), 1)
    data["citation_fidelity"] = round(data.get("grounded_claims", 0) / total, 4)
    data["phantom_citation_tags"] = phantom_citations
    return data