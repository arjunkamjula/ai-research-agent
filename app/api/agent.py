"""
app/api/agent.py

FastAPI route handlers for the research agent.

Routes:
  POST /agent/run      — run the agent on a question
  POST /agent/ingest   — add a document to the vector store
  GET  /agent/history  — last 20 agent runs
  GET  /health         — system health check
"""

import os
import time
from datetime import datetime
from collections import deque

import mlflow
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from app.agent.graph import run_agent
from app.schemas.request import AgentRequest, IngestRequest
from app.schemas.response import AgentResponse, ToolCallLog, HealthResponse
from app.tools.document_lookup import get_collection, ingest_document

load_dotenv()

router = Router = APIRouter()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# In-memory run history (last 20)
_history: deque = deque(maxlen=20)


@router.post("/agent/run", response_model=AgentResponse)
async def run(request: AgentRequest):
    """
    Run the research agent on a question.

    The agent will autonomously decide which tools to call
    (web search, document lookup, code execution, calculator)
    and iterate until it has a comprehensive answer.
    """
    start_time = time.time()
    run_id     = None

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("ai-research-agent")

        with mlflow.start_run(run_name=f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            mlflow.log_param("question",       request.question[:200])
            mlflow.log_param("max_iterations", request.max_iterations)
            mlflow.log_param("llm_provider",   LLM_PROVIDER)

            result = run_agent(
                question       = request.question,
                max_iterations = request.max_iterations,
            )

            latency_ms = round((time.time() - start_time) * 1000, 2)

            mlflow.log_metric("iterations",  result["iterations"])
            mlflow.log_metric("tokens_used", result["tokens_used"])
            mlflow.log_metric("tool_calls",  len(result["tool_calls"]))
            mlflow.log_metric("latency_ms",  latency_ms)

        tool_logs = [ToolCallLog(**tc) for tc in result["tool_calls"]]

        response = AgentResponse(
            answer        = result["answer"],
            tool_calls    = tool_logs,
            iterations    = result["iterations"],
            tokens_used   = result["tokens_used"],
            latency_ms    = latency_ms,
            mlflow_run_id = run_id,
        )

        _history.append({
            "question":   request.question,
            "answer":     result["answer"][:300],
            "tool_count": len(tool_logs),
            "iterations": result["iterations"],
            "tokens":     result["tokens_used"],
            "latency_ms": latency_ms,
            "timestamp":  datetime.now().isoformat(),
        })

        return response

    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/ingest")
async def ingest(request: IngestRequest):
    """Add a document to the vector store for document_lookup tool."""
    try:
        chunks_added = ingest_document(text=request.text, source=request.source)
        return {
            "status":       "ok",
            "chunks_added": chunks_added,
            "source":       request.source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/history")
async def history():
    """Return the last 20 agent runs."""
    return {"runs": list(reversed(list(_history)))}


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    try:
        collection    = get_collection()
        vector_count  = collection.count()
    except Exception:
        vector_count = -1

    return HealthResponse(
        status            = "ok",
        llm_provider      = LLM_PROVIDER,
        tools_available   = 4,
        vector_store_docs = vector_count,
    )
