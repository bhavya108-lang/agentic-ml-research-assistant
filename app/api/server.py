import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import your compiled LangGraph workflow
from app.graph.workflow import app as agent_workflow


# 1. Request and Expanded Response Schemas
class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    research: str
    research_sources: str
    local_evidence: str
    ml_analysis: str
    critique: str
    final_report: str
    revisions_triggered: int
    total_iterations: int


# 2. Initialize FastAPI App & CORS Middleware (for Phase 11 Frontend)
app = FastAPI(
    title="Agentic ML Research API",
    description="API for the Agentic ML Research Assistant with SSE streaming",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_initial_state(query: str) -> dict:
    return {
        "query": query.strip(),
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


# 3. Standard Blocking Endpoint (Returns full expanded state)
@app.post("/api/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = _build_initial_state(request.query)

    try:
        result = agent_workflow.invoke(initial_state)

        return ResearchResponse(
            query=result.get("query", request.query),
            research=result.get("research", ""),
            research_sources=result.get("research_sources", ""),
            local_evidence=result.get("local_evidence", ""),
            ml_analysis=result.get("ml_analysis", ""),
            critique=result.get("critique", ""),
            final_report=result.get("final_report", ""),
            revisions_triggered=result.get("revision_count", 0),
            total_iterations=result.get("iteration_count", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 4. Server-Sent Events (SSE) Streaming Endpoint
@app.post("/api/research/stream")
async def stream_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = _build_initial_state(request.query)

    def event_generator():
        final_state = dict(initial_state)
        start_time = time.perf_counter()
        step_start = start_time

        try:
            # Stream node-by-node updates as LangGraph executes
            for event in agent_workflow.stream(initial_state):
                now = time.perf_counter()
                for node_name, state_update in event.items():
                    duration = round(now - step_start, 3)
                    final_state.update(state_update)

                    payload = {
                        "event": "node_complete",
                        "node": node_name,
                        "latency_sec": duration,
                        "next_agent": final_state.get("next_agent", ""),
                        "iteration_count": final_state.get("iteration_count", 0),
                        "revision_count": final_state.get("revision_count", 0),
                        "update": state_update,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                step_start = time.perf_counter()

            # Emit final completion event with full state
            total_duration = round(time.perf_counter() - start_time, 3)
            complete_payload = {
                "event": "workflow_complete",
                "total_latency_sec": total_duration,
                "state": {
                    "query": final_state.get("query", ""),
                    "research": final_state.get("research", ""),
                    "research_sources": final_state.get("research_sources", ""),
                    "local_evidence": final_state.get("local_evidence", ""),
                    "ml_analysis": final_state.get("ml_analysis", ""),
                    "critique": final_state.get("critique", ""),
                    "final_report": final_state.get("final_report", ""),
                    "revisions_triggered": final_state.get("revision_count", 0),
                    "total_iterations": final_state.get("iteration_count", 0),
                },
            }
            yield f"data: {json.dumps(complete_payload)}\n\n"

        except Exception as e:
            error_payload = {"event": "error", "detail": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "ML Research Assistant API is running"}