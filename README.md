Agentic ML Research Assistant

An agentic AI research assistant designed for Machine Learning and Computer Vision research.

The system combines LLMs, LangGraph, RAG, hybrid retrieval, external research tools, multi-agent reasoning, critic-based revision, and Langfuse observability to investigate ML research questions and produce structured technical reports.

🚀 Project Overview

The goal of this project is to build an AI research assistant capable of:

Understanding Machine Learning and Computer Vision research questions

Retrieving relevant information from local research papers

Searching external web, paper, and dataset sources

Combining semantic and keyword-based retrieval

Using multiple specialized agents for research and ML analysis

Critically reviewing generated analysis

Automatically revising the analysis when necessary

Producing a structured final research report

Tracing agent execution and LLM calls with Langfuse

The project is being developed incrementally using a 12-phase architecture.

🏗️ Architecture

                         User Query
                              │
                              ▼
                    ┌──────────────────┐
                    │    Supervisor    │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       Research Agent     ML Agent        Critic
              │              │              │
              ▼              ▼              │
        ┌──────────┐   ┌──────────┐         │
        │ Local RAG│   │ ML       │         │
        │ Evidence │   │ Analysis │         │
        └──────────┘   └──────────┘         │
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       PASS / REVISE
                             │
                    ┌────────┴────────┐
                    │                 │
                   PASS             REVISE
                    │                 │
                    ▼                 ▼
              Final Report       ML Agent
                                      │
                                      ▼
                                   Critic
                                      │
                              bounded revisions
                                      │
                                      ▼
                                Final Report

🧭 Development Roadmap

PHASE 1  — LLM Layer
             Gemini + Groq
                 ↓
PHASE 2  — LangGraph Foundation
             State
             Nodes
             Edges
             Conditional Routing
                 ↓
PHASE 3  — Research Agent
             LLM reasoning
             Research planning
                 ↓
PHASE 4  — RAG
             PDF ingestion
             Chunking
             Embeddings
             Vector DB
             BM25
             Hybrid Retrieval
                 ↓
PHASE 5  — Tools
             Web Search
             Paper Search
             Dataset Search
                 ↓
PHASE 6  — Multi-Agent System
             Supervisor
             Research Agent
             Dataset Agent
             ML Agent
             Critic
                 ↓
PHASE 7  — Critic + Revision Loop
                 ↓
PHASE 8  — Langfuse
             Tracing
             Observability
                 ↓
PHASE 9  — Evaluation
             Retrieval Evaluation
             Agent Evaluation
             Answer Quality
                 ↓
PHASE 10 — FastAPI
                 ↓
PHASE 11 — Frontend
                 ↓
PHASE 12 — Deployment

✅ Current Progress

Phase 1 — LLM Layer ✅

Implemented support for:

Google Gemini

Groq

The active provider can be selected through .env:

LLM_PROVIDER=groq

or:

LLM_PROVIDER=gemini

The LLM layer is centralized in:

app/llm/provider.py

This keeps the rest of the application provider-independent.

Phase 2 — LangGraph Foundation ✅

Implemented the core LangGraph workflow with:

Shared ResearchState

Supervisor node

Research Agent node

ML Agent node

Critic node

Final Report node

Conditional routing

Workflow termination

Iteration limits

Workflow:

START
  ↓
Supervisor
  ↓
Research Agent
  ↓
Supervisor
  ↓
ML Agent
  ↓
Supervisor
  ↓
Critic
  ↓
Supervisor
  ↓
PASS / REVISE
  ↓
Final Report
  ↓
END

Phase 3 — Research Agent ✅

The Research Agent investigates the user's research question using:

Local RAG evidence

External research sources

LLM reasoning

Research analysis

It considers:

Computer Vision concepts

Model architectures

Datasets

Training considerations

Evaluation considerations

Research questions

Implemented in:

app/graph/workflow.py

Phase 4 — RAG System ✅

A local RAG pipeline has been implemented.

PDF ingestion

Research papers are loaded using PyPDF.

Example paper:

data/papers/yolo_v1.pdf

Chunking

Documents are split using recursive text splitting.

Embeddings

Gemini embeddings are used:

gemini-embedding-001

Vector Database

Chroma is used as the vector database:

data/chroma/

The Chroma database is excluded from Git.

BM25

Keyword-based retrieval is implemented using:

rank-bm25

Hybrid Retrieval

Semantic retrieval and BM25 keyword retrieval are combined using Reciprocal Rank Fusion (RRF):

Semantic Retrieval
       +
BM25 Keyword Retrieval
       ↓
Hybrid Retrieval

Phase 5 — External Research Tools ✅

The project includes a research tool architecture.

Web Search

Implemented using DDGS.

Retrieves:

Title

URL

Snippet

Paper Search

Implemented using the arXiv API.

Retrieves information such as:

Paper title

URL

Authors

Publication date

Description/snippet

Dataset Search

Implemented using the Hugging Face Datasets API.

Retrieves:

Dataset name

URL

Description

Tags

Phase 6 — Multi-Agent System 🚧

The current workflow contains:

Supervisor
Research Agent
ML Agent
Critic
Final Report

The roadmap also calls for a dedicated:

Dataset Agent

Currently, dataset searching is available through DatasetSearchTool and is invoked as part of the research workflow.

A dedicated Dataset Agent will be added as the multi-agent architecture is refined.

Phase 7 — Critic + Revision Loop ✅

The system contains an automated critic and revision loop.

The Critic evaluates:

Missing information

Unsupported claims

Contradictions

Technical mistakes

Insufficient reasoning

Missing experimental considerations

Claims unsupported by retrieved evidence

The Critic returns:

PASS

or:

REVISE

If REVISE is returned, the ML Agent is called again with the critic's feedback.

The revision loop is bounded using MAX_REVISIONS, preventing infinite loops.

The workflow also has an overall iteration safety limit.

📚 Evidence and Source Handling

The system distinguishes between local and external evidence.

Local RAG evidence

Uses:

[Evidence N]

External research sources

Uses:

[Research Source N]

Agents are instructed not to invent:

Evidence numbers

Sources

Papers

Datasets

Experimental results

State-of-the-art claims

The final report distinguishes retrieved evidence from the model's own technical reasoning.

📝 Final Research Report

The final report is structured as:

## Problem Understanding

## Research Findings

## ML Analysis

## Recommended Experimental Direction

## Limitations / Open Questions

## Evidence Sources

The evidence section separates:

### Local RAG Evidence

### External Research Sources

🔎 Source Normalization

External research results are normalized into a common structure:

{
    "source_type": ...,
    "title": ...,
    "url": ...,
    "content": ...,
    "metadata": ...
}

Implemented components include:

app/tools/source_normalizer.py
app/tools/source_formatter.py
app/tools/research_sources.py

📊 Observability

Langfuse integration has been added to the workflow.

The system currently supports tracing of the agent workflow and LLM calls.

Langfuse is integrated using:

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

The workflow attaches the Langfuse callback to the LangGraph execution.

Trace metadata includes information such as:

workflow
provider

and tags such as:

langgraph
multi-agent
computer-vision
research

Further observability improvements will be developed as the project progresses.

📁 Project Structure

agentic-ml-research-assistant/
│
├── app/
│   ├── agents/
│   │
│   ├── config/
│   │
│   ├── graph/
│   │   ├── workflow.py
│   │   └── visualize.py
│   │
│   ├── llm/
│   │   └── provider.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── bm25.py
│   │   ├── retriever.py
│   │   └── service.py
│   │
│   └── tools/
│       ├── research_tools.py
│       ├── web_search.py
│       ├── source_formatter.py
│       ├── paper_search.py
│       ├── dataset_search.py
│       ├── source_normalizer.py
│       └── research_sources.py
│
├── tests/
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_vector_store.py
│   ├── test_vector_search.py
│   ├── test_bm25.py
│   ├── test_hybrid.py
│   ├── test_rag_service.py
│   ├── test_research_tools.py
│   ├── test_web_search.py
│   ├── test_source_formatter.py
│   ├── test_paper_search.py
│   ├── test_dataset_search.py
│   ├── test_source_normalizer.py
│   └── test_research_sources.py
│
├── data/
│   ├── papers/
│   │   └── yolo_v1.pdf
│   └── chroma/
│
├── .env
├── .gitignore
├── requirements.txt
├── main.py
└── README.md

🧪 Testing

Dedicated tests have been created for major components:

PDF loading

Text splitting

Vector store

Vector search

BM25

Hybrid retrieval

RAG service

Research tools

Web search

Source formatting

Paper search

Dataset search

Source normalization

Research source aggregation

Tests are located in:

tests/

🔐 Environment Variables

Create a .env file:

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq

To use Gemini:

LLM_PROVIDER=gemini

To use Groq:

LLM_PROVIDER=groq

Never commit .env to GitHub.

The repository ignores:

.env
.venv/
__pycache__/
*.pyc
data/chroma/

▶️ Running the Project

Create and activate the virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the research workflow:

python -m app.graph.workflow

The current example query is:

Compare object detection models for traffic sign detection.

The workflow executes:

Supervisor
    ↓
Research Agent
    ↓
ML Agent
    ↓
Critic
    ↓
Revision loop if required
    ↓
Final Report

🧰 Technology Stack

Component

Technology

Language

Python

LLMs

Google Gemini, Groq

Agent Framework

LangGraph

LLM Framework

LangChain

RAG

LangChain

PDF Processing

PyPDF

Embeddings

Gemini Embeddings

Vector Database

Chroma

Keyword Retrieval

BM25

Web Search

DDGS

Paper Search

arXiv API

Dataset Search

Hugging Face API

Observability

Langfuse

Version Control

Git + GitHub

🛣️ Upcoming Development

Phase 8 — Observability

Planned improvements:

More detailed Langfuse tracing

Agent-level observability

Token usage tracking

Latency tracking

Failure analysis

Trace metadata

Phase 9 — Evaluation

Implement systematic evaluation for:

Retrieval

Retrieval quality

Relevant document retrieval

Hybrid retrieval performance

Agents

Research quality

ML reasoning quality

Critic effectiveness

Final Answers

Factual grounding

Citation correctness

Completeness

Answer quality

Phase 10 — FastAPI

Expose the research assistant through an API.

Planned architecture:

Frontend
    │
    ▼
FastAPI
    │
    ▼
LangGraph Workflow
    │
    ├── RAG
    ├── Research Tools
    ├── Research Agent
    ├── ML Agent
    └── Critic

Phase 11 — Frontend

Build a user interface for:

Entering research questions

Viewing agent progress

Viewing research findings

Viewing ML analysis

Viewing citations

Viewing final reports

Phase 12 — Deployment

Deploy the complete application.

Planned production architecture:

                User
                 │
                 ▼
             Frontend
                 │
                 ▼
              FastAPI
                 │
                 ▼
            LangGraph
          ┌──────┼──────┐
          ▼      ▼      ▼
         RAG   Tools   Agents
                 │
                 ▼
              LLM APIs
                 │
                 ▼
             Langfuse

🎯 Project Goal

The final objective is to build a complete production-style AI/ML research assistant rather than a simple LLM chatbot.

The finished system will combine:

LLMs
 +
LangGraph
 +
RAG
 +
Hybrid Retrieval
 +
External Research Tools
 +
Multi-Agent Reasoning
 +
Critic / Revision
 +
Observability
 +
Evaluation
 +
FastAPI
 +
Frontend
 +
Deployment

The project is being developed incrementally, with each major capability implemented, tested, and committed separately.

📌 Current Status

Core agentic research workflow: ✅ Working

LLM provider abstraction: ✅

LangGraph workflow: ✅

Local RAG + hybrid retrieval: ✅

Web/Paper/Dataset research tools: ✅

Critic + bounded revision loop: ✅

Langfuse integration: ✅

Dedicated Dataset Agent: 🚧 Planned

Evaluation: 🚧 Upcoming

FastAPI: 🚧 Upcoming

Frontend: 🚧 Upcoming

Deployment: 🚧 Upcoming