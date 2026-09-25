from app.tools.research_sources import search_research_sources
from app.rag.service import retrieve_evidence

from typing import TypedDict


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

from app.llm.provider import get_llm
llm = get_llm()

def get_response_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

        return "".join(text_parts)

    return str(content)

def supervisor(state: ResearchState):
    print("\n--- SUPERVISOR ---")

    iteration = state["iteration_count"]

    if iteration >= 6:
        print("Maximum iterations reached.")
        return {
            "next_agent": "final_report",
            "iteration_count": iteration + 1,
        }

    research_exists = bool(state["research"].strip())
    ml_exists = bool(state["ml_analysis"].strip())
    critique_exists = bool(state["critique"].strip())

    if not research_exists:
        decision = "research_agent"

    elif not ml_exists:
        decision = "ml_agent"

    elif not critique_exists:
        decision = "critic"

    elif state["critique"].strip().startswith("REVISE"):
        decision = "ml_agent"

    elif state["critique"].strip().startswith("PASS"):
        decision = "final_report"

    else:
        decision = "final_report"

    print(f"Supervisor decision: {decision}")

    return {
        "next_agent": decision,
        "iteration_count": iteration + 1,
    }

def research_agent(state: ResearchState):
    print("\n--- RESEARCH AGENT ---")

    evidence = retrieve_evidence(
        query=state["query"],
        top_n=5,
    )

     # External research sources
    research_sources = search_research_sources(
        query=state["query"],
        max_results=2,
    )

    prompt = f"""
You are the Research Agent in an Agentic ML Research Assistant.

Research question:
{state["query"]}

Retrieved evidence:
{evidence}

EXTERNAL RESEARCH SOURCES:
{research_sources}

Previous research:
{state["research"]}

Critic feedback:
{state["critique"]}

Analyze:

1. Relevant computer vision concepts
2. Model architectures
3. Datasets
4. Training considerations
5. Evaluation considerations
6. Important research questions

Evidence rules:

- Use local RAG evidence and external research sources
  to support factual claims.
- Cite local evidence using [Evidence N].
- Cite external research sources using [Research Source N].
- Do not invent evidence numbers.
- Do not invent sources, papers, datasets, page numbers,
  benchmarks, or numerical results.
- Clearly distinguish retrieved evidence from your own
  technical reasoning.
- If evidence does not establish something, say so clearly.

When external sources and local RAG evidence discuss the same
topic, compare them carefully rather than assuming they agree.

If previous research or critic feedback exists, improve it.
"""

    response = llm.invoke(prompt)

    return {
        "research": get_response_text(response),
        "research_sources": research_sources,
        "local_evidence": evidence
    }

def ml_agent(state: ResearchState):
    print("\n--- ML AGENT ---")

    evidence = state["local_evidence"]

    research_sources = state["research_sources"]

    prompt = f"""
You are the Machine Learning Agent in an AI/ML research assistant.

Original research question:
{state["query"]}

Research Agent findings:
{state["research"]}

Additional retrieved ML evidence:
{evidence}

EXTERNAL RESEARCH SOURCES:
{research_sources}

Previous ML analysis:
{state["ml_analysis"]}

Critic feedback:
{state["critique"]}

Analyze the machine learning aspects of the problem.

Discuss:

1. Suitable model architectures
2. Training strategy
3. Data requirements
4. Evaluation metrics
5. Potential challenges
6. Possible experiments
7. Important trade-offs

If critic feedback is provided, improve the previous ML analysis.

Important evidence rules:

- Use local RAG evidence to support claims when appropriate.
- Cite local RAG evidence using [Evidence N].
- Use external research sources when appropriate.
- Cite external sources using [Research Source N].
- Do not invent evidence numbers or research source numbers.
- Do not invent experimental results, papers, datasets, benchmarks,
  or numerical results.
- Clearly distinguish retrieved evidence from your own technical reasoning.

Do not invent experimental results or claim that an approach is
state-of-the-art without evidence.
"""

    response = llm.invoke(prompt)

    return {
        "ml_analysis": get_response_text(response),
        "critique": "",
    }

def critic(state: ResearchState):
    print("\n--- CRITIC ---")

    evidence = state["local_evidence"]

    research_sources = state["research_sources"]

    prompt = f"""
You are the Critic Agent for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{state["research"]}

ML analysis:
{state["ml_analysis"]}

Retrieved evidence for verification:
{evidence}

EXTERNAL RESEARCH SOURCES FOR VERIFICATION:
{research_sources}

Evaluate the analysis for:

1. Missing important information
2. Unsupported claims
3. Contradictions
4. Technical mistakes
5. Insufficient reasoning
6. Missing experimental considerations
7. Claims that are not supported by the retrieved evidence

Important evidence rules:

- Use the retrieved local RAG evidence when verifying factual claims.
- Cite local RAG evidence using [Evidence N].
- Use the external research sources when verifying claims based on
  web pages, papers, or datasets.
- Cite external sources using [Research Source N].
- Do not invent evidence numbers or research source numbers.
- Do not invent sources or experimental results.
- Distinguish unsupported claims from claims that are merely
  reasonable technical recommendations.
- If a claim is supported by external research but not by local
  RAG evidence, do not incorrectly mark it as unsupported.
- If external sources conflict with the analysis, explicitly identify
  the conflict.

First decide whether the analysis is acceptable.

Your response MUST start with exactly one of:

PASS

or

REVISE

Then explain your reasoning.

If REVISION is needed, clearly state what needs to be improved.
"""

    response = llm.invoke(prompt)

    return {
        "critique": get_response_text(response)
    }


def final_report(state: ResearchState):

    print("\n--- FINAL REPORT ---")

    prompt = f"""
You are the final report writer for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{state["research"]}

Machine learning analysis:
{state["ml_analysis"]}

Critic feedback:
{state["critique"]}

LOCAL RAG EVIDENCE:
{state["local_evidence"]}

EXTERNAL RESEARCH SOURCES:
{state["research_sources"]}

Create a concise but technically useful research report.

Structure it as:

## Problem Understanding

## Research Findings

## ML Analysis

## Recommended Experimental Direction

## Limitations / Open Questions

## Evidence Sources

Preserve all valid citations from the Research Agent and ML Agent.

Citation rules:

- Local RAG evidence MUST use [Evidence N].
- External web, paper, and dataset sources MUST use
  [Research Source N].
- Never replace [Research Source N] with [Evidence N].
- Never replace [Evidence N] with [Research Source N].
- Do not invent evidence numbers.
- Do not invent research source numbers.
- Do not invent sources, papers, datasets, page numbers,
  benchmarks, or experimental results.
- If a claim is based on the agent's own technical reasoning
  rather than retrieved evidence, clearly present it as
  technical reasoning rather than as a cited fact.

In the Evidence Sources section, create two subsections:

### Local RAG Evidence

For each [Evidence N] reference actually used in the report,
provide:
- Evidence number
- Source
- Page

### External Research Sources

For each [Research Source N] reference actually used in the report,
provide:
- Research source number
- Type
- Title
- URL

Only list sources that are actually referenced in the report.
Do not invent metadata.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": get_response_text(response)
    }

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
    }
)

graph.add_edge("research_agent", "supervisor")
graph.add_edge("ml_agent", "supervisor")
graph.add_edge("critic", "supervisor")

graph.add_edge("final_report", END)

app = graph.compile()


if __name__ == "__main__":

    result = app.invoke({
        "query": "Compare object detection models for traffic sign detection.",
        "research": "",
        "research_sources": "",
        "local_evidence": "",
        "ml_analysis": "",
        "critique": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0
    })

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print(result["final_report"])