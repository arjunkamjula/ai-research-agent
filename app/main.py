"""
app/main.py

AI Research Agent — FastAPI application entry point.

Run:
    uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.agent import router

load_dotenv()

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE",       "1")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ai-research-agent")
    yield


app = FastAPI(
    title       = "AI Research Agent",
    description = "Autonomous research agent with web search, document lookup, code execution, and calculator tools. Built with LangGraph ReAct loop.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(router)
