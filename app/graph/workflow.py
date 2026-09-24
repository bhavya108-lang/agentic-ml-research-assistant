from typing import TypedDict


class ResearchState(TypedDict):
    query: str
    research: str
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

    if state["iteration_count"] >= 6:
        print("Maximum iterations reached.")
        return {
            "next_agent": "final_report",
            "iteration_count": state["iteration_count"] + 1
        }

    research_exists = bool(state["research"].strip())
    ml_exists = bool(state["ml_analysis"].strip())
    critique_exists = bool(state["critique"].strip())

    critique_status = ""

    if critique_exists:
        critique_status = state["critique"][:1000]

    prompt = f"""
You are the Supervisor of an Agentic ML Research Assistant.

Original research question:
{state["query"]}

Current workflow status:

Research completed: {research_exists}
ML analysis completed: {ml_exists}
Critique completed: {critique_exists}

Critic feedback:
{critique_status}

Current iteration:
{state["iteration_count"]}

Available agents:

research_agent
ml_agent
critic
final_report

Rules:

- If research is not completed, choose research_agent.
- If research is completed but ML analysis is not completed,
  choose ml_agent.
- If research and ML analysis are completed but critique is not
  completed, choose critic.
- If critique starts with PASS, choose final_report.
- If critique starts with REVISE, choose the agent that needs
  improvement.
- Do not choose critic twice consecutively.
- If iteration reaches the limit, choose final_report.

Return ONLY one of:

research_agent
ml_agent
critic
final_report
"""

    response = llm.invoke(prompt)

    decision = get_response_text(response).strip().lower()

    allowed = {
        "research_agent",
        "ml_agent",
        "critic",
        "final_report"
    }

    if decision not in allowed:
        decision = "final_report"

    print(f"Supervisor decision: {decision}")

    return {
        "next_agent": decision,
        "iteration_count": state["iteration_count"] + 1
    }

def research_agent(state: ResearchState):
    print("\n--- RESEARCH AGENT ---")

    prompt = f"""
You are the Research Agent in an AI/ML research assistant.

User research question:
{state["query"]}

Previous research:
{state["research"]}

Critic feedback:
{state["critique"]}

Your task is to analyze the research question and provide:

1. What the problem is asking
2. Important computer vision / machine learning concepts
3. Relevant model families or approaches
4. Important considerations
5. Additional information that should be researched

If critic feedback is provided, improve the previous research
according to that feedback.

Do not invent papers, datasets, benchmarks, citations,
or numerical results.

Produce research analysis, not a final report.
"""

    response = llm.invoke(prompt)

    return {
        "research": get_response_text(response)
    }

def ml_agent(state: ResearchState):
    print("\n--- ML AGENT ---")

    prompt = f"""
You are the Machine Learning Agent in an AI/ML research assistant.

Original research question:
{state["query"]}

Research Agent findings:
{state["research"]}

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

Do not invent experimental results or claim that an approach is
state-of-the-art without evidence.
"""

    response = llm.invoke(prompt)

    return {
        "ml_analysis": get_response_text(response)
    }

def critic(state: ResearchState):
    print("\n--- CRITIC ---")

    prompt = f"""
You are the Critic Agent for an AI/ML research assistant.

Original question:
{state["query"]}

Research analysis:
{state["research"]}

ML analysis:
{state["ml_analysis"]}

Evaluate the analysis for:

1. Missing important information
2. Unsupported claims
3. Contradictions
4. Technical mistakes
5. Insufficient reasoning
6. Missing experimental considerations

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

Create a concise but technically useful research report.

Structure it as:

## Problem Understanding

## Research Findings

## ML Analysis

## Recommended Experimental Direction

## Limitations / Open Questions

Do not invent sources, experimental results, or citations.
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