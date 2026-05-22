"""app/schemas/response.py"""
from pydantic import BaseModel
from typing import Optional


class ToolCallLog(BaseModel):
    tool:   str
    input:  str
    output: str


class AgentResponse(BaseModel):
    answer:        str
    tool_calls:    list[ToolCallLog]
    iterations:    int
    tokens_used:   int
    latency_ms:    float
    mlflow_run_id: Optional[str] = None
    error:         Optional[str] = None


class HealthResponse(BaseModel):
    status:           str
    llm_provider:     str
    tools_available:  int
    vector_store_docs: int
