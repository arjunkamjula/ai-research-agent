"""app/schemas/request.py"""
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question:       str = Field(..., min_length=3)
    max_iterations: int = Field(8, ge=1, le=20)


class IngestRequest(BaseModel):
    text:   str = Field(..., min_length=10)
    source: str = Field("manual")
