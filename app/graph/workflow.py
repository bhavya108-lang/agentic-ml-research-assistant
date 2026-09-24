from typing import TypedDict


class ResearchState(TypedDict):
    query: str
    research: str
    ml_analysis: str
    critique: str
    final_report: str
    status: str

def supervisor(state: ResearchState):
    print("\n--- SUPERVISOR ---")

    return {
        "status": "research"
    }


def research_agent(state: ResearchState):
    print("\n--- RESEARCH AGENT ---")

    return {
        "research": "Initial research findings."
    }


def ml_agent(state: ResearchState):
    print("\n--- ML AGENT ---")

    return {
        "ml_analysis": "Initial machine learning analysis."
    }


def critic(state: ResearchState):
    print("\n--- CRITIC ---")

    return {
        "critique": "The response looks good.",
        "status": "approved"
    }


def final_report(state: ResearchState):
    print("\n--- FINAL REPORT ---")

    return {
        "final_report": (
            f"Research:\n{state['research']}\n\n"
            f"ML Analysis:\n{state['ml_analysis']}\n\n"
            f"Critique:\n{state['critique']}"
        )
    }


from langgraph.graph import StateGraph, START, END


graph = StateGraph(ResearchState)

graph.add_node("supervisor", supervisor)
graph.add_node("research_agent", research_agent)
graph.add_node("ml_agent", ml_agent)
graph.add_node("critic", critic)
graph.add_node("final_report", final_report)

graph.add_edge(START, "supervisor")

graph.add_edge("supervisor", "research_agent")

graph.add_edge("research_agent", "ml_agent")

graph.add_edge("ml_agent", "critic")

graph.add_edge("critic", "final_report")

graph.add_edge("final_report", END)

app = graph.compile()


if __name__ == "__main__":

    result = app.invoke({
        "query": "Compare object detection models for traffic sign detection.",
        "research": "",
        "ml_analysis": "",
        "critique": "",
        "final_report": "",
        "status": ""
    })

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print(result["final_report"])