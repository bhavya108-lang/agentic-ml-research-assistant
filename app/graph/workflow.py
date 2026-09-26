import os
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
MAX_SOURCES_CHARS = 1500      # external research sources block (search, unbounded)
MAX_ANALYSIS_CHARS = 5000     # research / ml_analysis re-fed elsewhere — set
                              # above the ~4800-char ceiling of a 1200-token
                              # response so it's effectively a safety cap, not
                              # a truncation that fires in normal operation
MAX_CRITIQUE_CHARS = 3000     # critic feedback fed back into revisions — set
                              # above the ~2800-char ceiling of a 700-token
                              # critic response for the same reason
MAX_ITERATIONS = 8            # hard safety cap on the supervisor loop
MAX_REVISIONS = 2             # ml_agent <-> critic revision loop cap


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


from app.llm.provider import get_llm

# Per-node output budgets. The critic only needs a short verdict, so giving
# it the same 1200-token ceiling as the analysis agents wastes output tokens
# on nearly every supervisor loop. final_report gets more room since it has
# to synthesize everything into six sections plus a sourced bibliography.
llm = get_llm(max_tokens=1200)              # research_agent, ml_agent
llm_critic = get_llm(max_tokens=900)         # critic — combined with
                                              # reasoning_effort="low" in
                                              # provider.py; if that param
                                              # isn't supported by your
                                              # installed langchain_groq,
                                              # this larger budget is the
                                              # fallback safety margin
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
- Separate retrieved evidence from your own reasoning; say so if evidence is thin.
"""


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
    critique_exists = bool(state["critique"].strip())

    if not research_exists:
        decision = "research_agent"

    elif not ml_exists:
        decision = "ml_agent"

    elif not critique_exists:
        # FIX: this branch used to jump straight to "final_report", which
        # meant the critic node was never reachable through routing and all
        # the PASS/REVISE logic below it was dead code. Now it actually
        # routes to the critic first.
        decision = "critic"

    elif state["critique"].strip().startswith("REVISE"):
        if revision_count < MAX_REVISIONS:
            revision_count += 1
            print(f"Revision requested. Revision attempt: {revision_count}/{MAX_REVISIONS}")
            decision = "ml_agent"
        else:
            print("Maximum revision attempts reached. Proceeding to final report.")
            decision = "final_report"

    elif state["critique"].strip().startswith("PASS"):
        decision = "final_report"

    else:
        print("Critic returned an unexpected response. Proceeding to final report.")
        decision = "final_report"

    print(f"Supervisor decision: {decision}")

    return {
        "next_agent": decision,
        "iteration_count": iteration + 1,
        "revision_count": revision_count,
    }


def research_agent(state: ResearchState):
    print("\n--- RESEARCH AGENT ---")

    # Evidence is only fetched once per run (supervisor never routes back
    # here after research exists), so there's no wasted re-fetching.
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
        "critique": "",
    }


def critic(state: ResearchState):
    print("\n--- CRITIC ---")

    prompt = f"""You are the Critic Agent for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{truncate(state["research"], MAX_ANALYSIS_CHARS)}

ML analysis:
{truncate(state["ml_analysis"], MAX_ANALYSIS_CHARS)}

Evidence for verification:
{truncate(state["local_evidence"], MAX_EVIDENCE_CHARS)}

External research sources for verification:
{truncate(state["research_sources"], MAX_SOURCES_CHARS)}

Evaluate for: missing information, unsupported claims, contradictions,
technical mistakes, insufficient reasoning, missing experimental
considerations, and claims unsupported by the retrieved evidence.
{EVIDENCE_RULES}
Distinguish unsupported claims from reasonable technical recommendations.
If a claim is supported by external research but not local evidence, don't
mark it unsupported. Flag any conflicts between external sources and the analysis.

Your response MUST start with exactly one of: PASS or REVISE
Then explain your reasoning concisely. If REVISE, state exactly what needs improving.
"""

    critique_text = invoke_safely(
        llm_critic,
        prompt,
        "Critic",
        fallback_text="PASS (critic returned no content; auto-approved to avoid a stall)",
    )

    return {"critique": critique_text}


def final_report(state: ResearchState):
    print("\n--- FINAL REPORT ---")

    prompt = f"""You are the final report writer for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{truncate(state["research"], MAX_ANALYSIS_CHARS)}

Machine learning analysis:
{truncate(state["ml_analysis"], MAX_ANALYSIS_CHARS)}
{section("Critic feedback", state["critique"], MAX_CRITIQUE_CHARS)}
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