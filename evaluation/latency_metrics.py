import time
from app.graph.workflow import app

def measure_workflow_latency(query: str) -> dict:
    initial_state = {
        "query": query,
        "research": "",
        "research_sources": "",
        "local_evidence": "",
        "ml_analysis": "",
        "critique": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
        "revision_count": 0,
    }

    node_timings = []
    start_total = time.perf_counter()
    step_start = start_total
    final_state = dict(initial_state)

    for event in app.stream(initial_state):
        now = time.perf_counter()
        for node_name, state_update in event.items():
            duration = round(now - step_start, 3)
            node_timings.append({"node": node_name, "latency_sec": duration})
            final_state.update(state_update)
        step_start = time.perf_counter()

    total_latency = round(time.perf_counter() - start_total, 3)
    
    # Calculate revision loop overhead (any ml_agent or critic calls after their first pass)
    ml_calls = [t["latency_sec"] for t in node_timings if t["node"] == "ml_agent"]
    critic_calls = [t["latency_sec"] for t in node_timings if t["node"] == "critic"]
    revision_overhead = round(sum(ml_calls[1:]) + sum(critic_calls[1:]), 3)

    return {
        "total_latency_sec": total_latency,
        "revision_overhead_sec": revision_overhead,
        "revisions_triggered": final_state.get("revision_count", 0),
        "node_breakdown": node_timings,
        "final_state": final_state,
    }