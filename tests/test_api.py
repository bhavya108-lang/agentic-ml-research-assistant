import pytest
from fastapi.testclient import TestClient
from app.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_research_endpoint_validation():
    # Test empty query validation
    response = client.post("/api/research", json={"query": "   "})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]

def test_stream_research_endpoint_validation():
    # Test empty query validation for stream
    response = client.post("/api/research/stream", json={"query": ""})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]

def test_stream_research_headers():
    # Check that stream endpoint correctly initializes a text/event-stream connection
    with client.stream("POST", "/api/research/stream", json={"query": "What is YOLO?"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")