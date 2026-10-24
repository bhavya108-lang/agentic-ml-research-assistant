import json
import re
import os
import unicodedata
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.tools.research_sources import search_research_sources
from app.rag.service import retrieve_evidence

from typing import TypedDict, Optional

langfuse = get_client()
langfuse_handler = CallbackHandler()

# ---------------------------------------------------------------------------
# Token-budget controls
#
# research/ml_analysis/critique are already bounded by their own max_tokens
# at generation time (1200 / 1200 / 700 tokens ≈ 4800 / 4800 / 2800 chars).
# Truncating them AGAIN below that on the way into the next prompt doesn't
# save real tokens (they're already capped) — it just deletes content the
# critic or the revising agent actually needs, which was very likely why
# REVISE kept firing for the same unaddressed reasons: the critic couldn't
# see the whole analysis, and ml_agent couldn't see the whole critique.
# So these caps only apply to the genuinely unbounded, verbose blocks:
# raw retrieved evidence and external search results.
# ---------------------------------------------------------------------------
MAX_EVIDENCE_CHARS = 2500     # local RAG evidence block (retrieval, unbounded)
MAX_SOURCES_CHARS = 3500      # external research sources block (search, unbounded)
MAX_ANALYSIS_CHARS = 5000     # research / ml_analysis re-fed elsewhere — set
                              # above the ~4800-char ceiling of a 1200-token
                              # response so it's effectively a safety cap, not
                              # a truncation that fires in normal operation
MAX_CRITIQUE_CHARS = 3000     # critic feedback fed back into revisions — set
                              # above the ~2800-char ceiling of a 700-token
                              # critic response for the same reason
MAX_ITERATIONS = 10           # hard safety cap on the supervisor loop (research + ml + 3 critic
                              # passes + 2 revisions + final fits comfortably)
MAX_REVISIONS = 2             # revision loop cap (a revision can only remove/relabel claims)
FIDELITY_THRESHOLD = 0.80     # min share of factual claims backed by a VERIFIED evidence quote
MIN_QUOTE_WORDS = 4           # shorter "quotes" are too easy to match by accident
MAX_CLAIMS = 20               # claims the critic extracts per pass (keeps its JSON bounded)


class ResearchState(TypedDict):
    query: str
    research: str
    research_sources: str
    local_evidence: str
    ml_analysis: str
    critique: str
    final_report: str
    next_agent: str
    iteration_count: int
    revision_count: int
    critic_verdict: str          # "", "pass", "revise", "unverified"
    fidelity: Optional[float]    # verified factual claims / total factual claims
    unsupported_claims: list     # [{"claim": str, "origin": "research" | "ml"}]


from app.llm.provider import get_llm

# Per-node output budgets. The critic only needs a short verdict, so giving
# it the same 1200-token ceiling as the analysis agents wastes output tokens
# on nearly every supervisor loop. final_report gets more room since it has
# to synthesize everything into six sections plus a sourced bibliography.
llm = get_llm(max_tokens=1200)              # research_agent, ml_agent
llm_critic = get_llm(max_tokens=3000)        # critic — emits one JSON record (with an evidence
                                              # quote) per claim, far longer than a PASS/REVISE
                                              # verdict. At 900 tokens the JSON gets cut off,
                                              # parsing fails, and every run becomes "unverified".
                                              # Headroom also covers hidden reasoning tokens.
llm_report = get_llm(max_tokens=1800)       # final_report


def get_response_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return str(content)


import time


def invoke_safely(
    llm_instance,
    prompt: str,
    node_name: str,
    fallback_text: str,
    retries: int = 2,
    backoff_seconds: int = 5,
) -> str:
    """Call an LLM with two safety nets, both aimed at not wasting API quota:

    1. Retry on transient errors (e.g. a Groq rate-limit hiccup) instead of
       crashing the whole run and losing every call already spent.
    2. If the model ever returns empty text (a reasoning model can burn its
       whole token budget on hidden reasoning and leave nothing visible),
       fall back to a placeholder instead of leaving the state field blank.
       A blank field makes the supervisor think this node hasn't run yet,
       so it just keeps calling it again until MAX_ITERATIONS is hit —
       spending real API calls for nothing.
    """
    last_err = None

    for attempt in range(retries + 1):
        try:
            response = llm_instance.invoke(prompt)
            text = get_response_text(response).strip()

            if not text:
                print(f"{node_name} returned empty content. Using fallback text.")
                return fallback_text

            return text

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = backoff_seconds * (attempt + 1)
                print(f"{node_name} call failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)

    print(f"{node_name} failed after {retries + 1} attempts ({last_err}). Using fallback text.")
    return fallback_text


def truncate(text: str, max_chars: int) -> str:
    """Hard-cap a text block so it can't blow the prompt token budget."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated for length]"


def section(title: str, content: str, max_chars: Optional[int] = None) -> str:
    """Only emit a prompt section when there's actually something to say.
    Skips several hundred wasted tokens per call on empty state fields
    (e.g. 'Critic feedback:\n\n' when there is no critique yet)."""
    content = (content or "").strip()
    if not content:
        return ""
    if max_chars:
        content = truncate(content, max_chars)
    return f"\n{title}:\n{content}\n"


EVIDENCE_RULES = """
Evidence rules:
- Cite local evidence as [Evidence N], external sources as [Research Source N].
- Never invent evidence/source numbers, papers, datasets, or results.
- State as fact ONLY what the retrieved evidence or sources explicitly say. Do not add
  facts from background knowledge, even if you believe they are true.
- Prefix your own proposals, hypotheses and design ideas with "Suggestion:" so they
  are never mistaken for evidence-backed findings.
- If evidence is thin, say so instead of filling the gap.
"""

REVISION_RULES = """
Revision rules (the critic found unsupported claims):
- For every claim the critic lists: delete it, rewrite it as a clearly labelled
  "Suggestion:", or replace it with wording the evidence explicitly supports.
- Do NOT add new factual claims while revising. A revision cannot create evidence.
"""


def revision_rules(state) -> str:
    """Only include revision instructions when there is critic feedback to act on."""
    return REVISION_RULES if (state.get("critique") or "").strip() else ""


def supervisor(state: ResearchState):
    print("\n--- SUPERVISOR ---")

    iteration = state["iteration_count"]
    revision_count = state["revision_count"]

    # Safety limit for the entire workflow
    if iteration >= MAX_ITERATIONS:
        print("Maximum iterations reached.")
        return {
            "next_agent": "final_report",
            "iteration_count": iteration + 1,
            "revision_count": revision_count,
        }

    research_exists = bool(state["research"].strip())
    ml_exists = bool(state["ml_analysis"].strip())
    # Route on the structured verdict, not on a string prefix of the critique.
    verdict = state.get("critic_verdict", "")

    if not research_exists:
        decision = "research_agent"

    elif not ml_exists:
        decision = "ml_agent"

    elif not verdict:
        decision = "critic"

    elif verdict == "revise":
        if revision_count < MAX_REVISIONS:
            revision_count += 1
            # Send the revision to whichever agent produced the unsupported claims.
            origins = {c.get("origin") for c in state.get("unsupported_claims", [])}
            decision = "research_agent" if "research" in origins else "ml_agent"
            print(f"Revision requested. Revision attempt: {revision_count}/{MAX_REVISIONS} -> {decision}")
        else:
            print("Maximum revision attempts reached. Proceeding to final report with warnings.")
            decision = "final_report"

    elif verdict == "unverified":
        # A broken critic is not a pass, but revising can't fix it either.
        # Proceed, and let final_report mark the output as unverified.
        print("Critic could not verify the analysis. Proceeding to final report with a warning.")
        decision = "final_report"

    else:  # "pass" (or anything unexpected)
        decision = "final_report"

    print(f"Supervisor decision: {decision}")

    return {
        "next_agent": decision,
        "iteration_count": iteration + 1,
        "revision_count": revision_count,
    }


def research_agent(state: ResearchState):
    print("\n--- RESEARCH AGENT ---")

    # Evidence is fetched only on the first pass. On a revision we reuse exactly what
    # the critic verified against; re-fetching could change the evidence and make the
    # critic's quotes un-checkable.
    if state["research"].strip():
        evidence = state["local_evidence"]
        research_sources = state["research_sources"]
    else:
        evidence = retrieve_evidence(query=state["query"], top_n=5)
        research_sources = search_research_sources(query=state["query"], max_results=2)

    prompt = f"""You are the Research Agent in an Agentic ML Research Assistant.

Research question:
{state["query"]}

Retrieved evidence:
{truncate(evidence, MAX_EVIDENCE_CHARS)}

External research sources:
{truncate(research_sources, MAX_SOURCES_CHARS)}
{section("Previous research (revise/improve this)", state["research"], MAX_ANALYSIS_CHARS)}{section("Critic feedback", state["critique"], MAX_CRITIQUE_CHARS)}
Analyze: computer vision concepts, model architectures, datasets,
training considerations, evaluation considerations, and key research questions.
{EVIDENCE_RULES}
When external sources and local evidence discuss the same topic, compare them
rather than assuming agreement.
{revision_rules(state)}
"""

    response_text = invoke_safely(
        llm,
        prompt,
        "Research agent",
        fallback_text=(
            "Research agent returned no content this attempt. "
            "Treat prior research/evidence as the best available and proceed with caution."
        ),
    )

    return {
        "research": response_text,
        "research_sources": research_sources,
        "local_evidence": evidence,
        # Force a fresh critic pass on the revised text.
        "critique": "",
        "critic_verdict": "",
        "fidelity": None,
        "unsupported_claims": [],
    }


def ml_agent(state: ResearchState):
    print("\n--- ML AGENT ---")

    prompt = f"""You are the Machine Learning Agent in an AI/ML research assistant.

Original research question:
{state["query"]}

Research Agent findings:
{truncate(state["research"], MAX_ANALYSIS_CHARS)}

Additional retrieved ML evidence:
{truncate(state["local_evidence"], MAX_EVIDENCE_CHARS)}

External research sources:
{truncate(state["research_sources"], MAX_SOURCES_CHARS)}
{section("Previous ML analysis (revise/improve this)", state["ml_analysis"], MAX_ANALYSIS_CHARS)}{section("Critic feedback", state["critique"], MAX_CRITIQUE_CHARS)}
Discuss: suitable model architectures, training strategy, data requirements,
evaluation metrics, potential challenges, possible experiments, and trade-offs.
{EVIDENCE_RULES}
Do not invent experimental results or claim state-of-the-art without evidence.
{revision_rules(state)}
"""

    response_text = invoke_safely(
        llm,
        prompt,
        "ML agent",
        fallback_text=(
            "ML agent returned no content this attempt. "
            "Treat prior ML analysis as the best available and proceed with caution."
        ),
    )

    return {
        "ml_analysis": response_text,
        # Force a fresh critic pass on the revised text.
        "critique": "",
        "critic_verdict": "",
        "fidelity": None,
        "unsupported_claims": [],
    }


def _norm(s: str) -> str:
    """Normalise text so quote matching ignores case, whitespace, ligatures, dash/quote styles."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    return re.sub(r"\s+", " ", s).strip()


def _quote_in_evidence(quote: str, norm_evidence: str) -> bool:
    """True only if the quote (split on ellipses) really occurs in the evidence."""
    parts = [p.strip(" .\"'") for p in re.split(r"\.\.\.|\u2026", _norm(quote))]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.split()) for p in parts) < MIN_QUOTE_WORDS:
        return False
    return all(p in norm_evidence for p in parts)


def _unverified(reason: str) -> dict:
    # Fail closed: a critic that could not run is "unverified", never a silent pass.
    return {
        "critique": f"UNVERIFIED: {reason}",
        "critic_verdict": "unverified",
        "fidelity": None,
        "unsupported_claims": [],
    }


def critic(state: ResearchState):
    print("\n--- CRITIC ---")

    # Build the evidence ONCE and use the same string for the prompt and for quote
    # verification, so truncation can't cause a quote to look fabricated.
    evidence_text = "\n\n".join(
        block
        for block in (
            truncate(state["local_evidence"], MAX_EVIDENCE_CHARS),
            truncate(state["research_sources"], MAX_SOURCES_CHARS),
        )
        if block
    )

    if not evidence_text:
        # Nothing to check against; skip the LLM call instead of failing every claim.
        print("Critic skipped: no evidence retrieved.")
        return _unverified("no evidence was retrieved, so claims cannot be checked.")

    prompt = f"""You are a strict fact-checker for an AI/ML research assistant.

Original question:
{state["query"]}

ANALYSES TO VERIFY

[RESEARCH]
{truncate(state["research"], MAX_ANALYSIS_CHARS)}

[ML]
{truncate(state["ml_analysis"], MAX_ANALYSIS_CHARS)}

EVIDENCE (the ONLY source of truth)
{evidence_text}

INSTRUCTIONS
1. Extract up to {MAX_CLAIMS} of the most important distinct claims from the analyses, one record each.
   Set "origin" to "research" or "ml" depending on which analysis made the claim.
2. Set "type" to "fact" for statements about methods, numbers, results, datasets or comparisons.
   Use "suggestion" ONLY for statements that are explicitly the author's own proposal
   (prefixed "Suggestion:" or phrased as a recommendation). When unsure, use "fact".
3. For each "fact", set "supported" to true ONLY if the EVIDENCE explicitly states it.
   Copy a verbatim quote (at most 25 words) from the EVIDENCE into "evidence_quote".
   If you cannot quote supporting text, set "supported" to false and leave the quote empty.
4. Never use your own background knowledge, even for claims that are true in general.
   A claim being correct is not the same as it being supported.

Return ONLY valid JSON, no markdown:
{{
  "claims": [
    {{"claim": "...", "origin": "research", "type": "fact", "supported": true, "evidence_quote": "..."}}
  ],
  "explanation": "<one or two sentences>"
}}"""

    data, last_raw = None, ""
    for _ in range(2):  # one retry on unparseable output
        last_raw = invoke_safely(llm_critic, prompt, "Critic", fallback_text="")
        match = re.search(r"\{.*\}", last_raw, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue

    if data is None or not isinstance(data.get("claims"), list):
        print(f"Critic output could not be parsed. Raw output: {last_raw[:300]}")
        return _unverified("critic output could not be parsed.")

    norm_evidence = _norm(evidence_text)
    facts, unsupported = [], []
    for c in data["claims"]:
        if not isinstance(c, dict) or c.get("type") != "fact":
            continue
        facts.append(c)
        # Supported only if the critic says so AND its quote really exists in the evidence.
        claimed = c.get("supported") in (True, "true", "True")
        if not (claimed and _quote_in_evidence(c.get("evidence_quote", ""), norm_evidence)):
            origin = c.get("origin") if c.get("origin") in ("research", "ml") else "ml"
            unsupported.append({"claim": c.get("claim", ""), "origin": origin})

    total = len(facts)
    if total == 0:
        # A research analysis with zero factual claims means extraction failed.
        return _unverified("critic extracted no factual claims to check.")

    verified = total - len(unsupported)
    fidelity = verified / total
    explanation = data.get("explanation", "")
    print(f"Critic fidelity: {fidelity:.2f} ({verified}/{total} factual claims verified)")

    unsupported_lines = "\n".join(f"- [{u['origin']}] {u['claim']}" for u in unsupported)

    if fidelity < FIDELITY_THRESHOLD:
        verdict = "revise"
        critique_text = (
            f"REVISE\n"
            f"Fidelity {fidelity:.2f} ({verified}/{total} verified; threshold {FIDELITY_THRESHOLD:.2f}).\n"
            f"{explanation}\n"
            f"Unsupported claims (delete, relabel as 'Suggestion:', or support with real evidence):\n"
            f"{unsupported_lines}"
        )
    else:
        verdict = "pass"
        critique_text = f"PASS\nFidelity {fidelity:.2f} ({verified}/{total} verified).\n{explanation}"
        if unsupported:
            critique_text += f"\nStill unverified (must not appear as findings):\n{unsupported_lines}"

    return {
        "critique": critique_text,
        "critic_verdict": verdict,
        "fidelity": fidelity,
        "unsupported_claims": unsupported,
    }


def final_report(state: ResearchState):
    print("\n--- FINAL REPORT ---")

    verdict = state.get("critic_verdict", "")
    fidelity = state.get("fidelity")
    unsupported = state.get("unsupported_claims") or []

    # Be explicit about how well this report was verified.
    if verdict == "pass":
        status = f"PASSED (fidelity {fidelity:.2f})."
    elif verdict == "revise":
        status = (
            f"FAILED after {MAX_REVISIONS} revisions (fidelity {fidelity:.2f}); "
            "some claims remain unverified."
        )
    else:
        status = "NOT COMPLETED: the claims in this report were not verified against the evidence."

    gaps = []
    if not (state["local_evidence"] or "").strip():
        gaps.append("No local RAG evidence was retrieved.")
    if not (state["research_sources"] or "").strip():
        gaps.append("No external research sources were retrieved.")
    verification = status + (" " + " ".join(gaps) if gaps else "")

    # Claims that failed verification must not come back in as findings, even when
    # the overall verdict is PASS (a pass at 0.85 still leaves 15% unverified).
    unsupported_block = ""
    if unsupported:
        listed = "\n".join(f"- {u.get('claim', '')}" for u in unsupported)
        unsupported_block = f"""
Claims the critic could NOT verify against the evidence. Do NOT state them as findings.
Omit them, or at most list them under "Limitations / Open Questions" as unverified:
{listed}
"""

    prompt = f"""You are the final report writer for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{truncate(state["research"], MAX_ANALYSIS_CHARS)}

Machine learning analysis:
{truncate(state["ml_analysis"], MAX_ANALYSIS_CHARS)}

Verification status: {verification}
{unsupported_block}
Local RAG evidence:
{truncate(state["local_evidence"], MAX_EVIDENCE_CHARS)}

External research sources:
{truncate(state["research_sources"], MAX_SOURCES_CHARS)}

Create a concise but technically useful research report structured as:

## Problem Understanding
## Research Findings
## ML Analysis
## Recommended Experimental Direction
## Limitations / Open Questions
## Evidence Sources

Grounding rules (strict):
- Do NOT introduce any factual claim that is not already in the analyses or evidence above.
  You are synthesizing, not adding knowledge.
- Every factual statement must carry a citation. Your own proposals belong in
  "Recommended Experimental Direction" and must start with "Suggestion:".
- Begin "Limitations / Open Questions" with the verification status above, unless it PASSED
  with no unverified claims.

Preserve all valid citations from the Research and ML agents.
- Local RAG evidence MUST use [Evidence N]; external sources MUST use [Research Source N].
- Never swap the two citation types. Never invent numbers, sources, or results.
- Present the agent's own technical reasoning as reasoning, not as a cited fact.

In "Evidence Sources", list only references actually used in the report, split into:
### Local RAG Evidence — number, source, page
### External Research Sources — number, type, title, URL
"""

    report_text = invoke_safely(
        llm_report,
        prompt,
        "Final report",
        fallback_text=(
            "Final report generation returned no content. "
            "See the research, ML analysis, and critique fields above for the underlying findings."
        ),
    )

    return {"final_report": report_text}


def route_from_supervisor(state: ResearchState):
    return state["next_agent"]


from langgraph.graph import StateGraph, START, END

graph = StateGraph(ResearchState)

graph.add_node("supervisor", supervisor)
graph.add_node("research_agent", research_agent)
graph.add_node("ml_agent", ml_agent)
graph.add_node("critic", critic)
graph.add_node("final_report", final_report)

graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research_agent": "research_agent",
        "ml_agent": "ml_agent",
        "critic": "critic",
        "final_report": "final_report",
    },
)

graph.add_edge("research_agent", "supervisor")
graph.add_edge("ml_agent", "supervisor")
graph.add_edge("critic", "supervisor")
graph.add_edge("final_report", END)

app = graph.compile()


if __name__ == "__main__":
    with propagate_attributes(
        trace_name="Agentic ML Research Assistant",
        tags=["langgraph", "multi-agent", "computer-vision", "research"],
        metadata={
            "workflow": "agentic-ml-research-assistant",
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
        },
    ):
        result = app.invoke(
            {
                "query": "Compare object detection models for traffic sign detection.",
                "research": "",
                "research_sources": "",
                "local_evidence": "",
                "ml_analysis": "",
                "critique": "",
                "final_report": "",
                "next_agent": "",
                "iteration_count": 0,
                "revision_count": 0,
                "critic_verdict": "",
                "fidelity": None,
                "unsupported_claims": [],
            },
            config={
                "callbacks": [langfuse_handler],
                "run_name": "traffic-sign-detection-research",
            },
        )

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")
    print(result["final_report"])

    langfuse.flush()