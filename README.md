# AI Research Agent

An autonomous research agent that answers complex questions by deciding which tools to call, in what order, and how to synthesize results into a coherent response. Built with a ReAct (Reasoning + Acting) loop, LangGraph state machine, and served via FastAPI.

The agent reasons about a question, selects from available tools, executes them, observes the result, and repeats until it has enough information to answer. Each step is logged with reasoning traces so you can see exactly how the agent reached its conclusion.

---

## What it does

Give the agent a question like "What are the latest developments in multimodal LLMs and write me a Python script to call one?" and it will:

1. Search the web for recent information
2. Retrieve relevant documents from its vector store
3. Write and execute Python code if needed
4. Compute or calculate if the question involves numbers
5. Synthesize everything into a structured answer with citations

The agent decides the tool sequence autonomously — it is not a fixed pipeline.

---

## Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph — state machine with conditional edges |
| LLM | Groq (Llama-3.3-70b) / OpenAI GPT-4o — one env var swap |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, free) |
| Vector store | ChromaDB (persistent local) |
| Web search | DuckDuckGo Search API (free, no key needed) |
| Code execution | Python subprocess sandbox |
| API | FastAPI + Uvicorn |
| Monitoring | MLflow — logs every agent run, tool calls, token usage |
| Validation | Pydantic v2 |

---

## Architecture

```
POST /agent/run
      │
      ▼
  AgentState initialized
  {question, messages, tool_calls, iterations, final_answer}
      │
      ▼
┌─────────────────────────────────────────┐
│           LangGraph State Machine        │
│                                         │
│  ┌──────────┐                           │
│  │  REASON  │ ← LLM decides:            │
│  │          │   use a tool OR answer    │
│  └────┬─────┘                           │
│       │                                 │
│  ┌────▼─────────────────────────────┐   │
│  │  ROUTE (conditional edge)        │   │
│  │  tool_call detected? → EXECUTE   │   │
│  │  no tool needed?    → ANSWER     │   │
│  │  max iterations?    → ANSWER     │   │
│  └────┬──────────────────────┬──────┘   │
│       │                      │          │
│  ┌────▼─────┐           ┌────▼─────┐   │
│  │ EXECUTE  │           │ ANSWER   │   │
│  │ tool     │           │ synthesize│  │
│  │ observe  │           │ + return  │   │
│  └────┬─────┘           └──────────┘   │
│       │                                 │
│       └──────────────► REASON (loop)   │
└─────────────────────────────────────────┘
      │
      ▼
  MLflow run logged
  Response returned
```

---

## Tools

| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo search — returns top 5 results with snippets |
| `document_lookup` | Semantic search over ChromaDB vector store |
| `python_executor` | Runs Python code in a subprocess sandbox, returns stdout |
| `calculator` | Safe arithmetic evaluation — no exec(), pure math |

The agent has access to all four simultaneously and decides which to call.

---

## Setup

```bash
git clone <repo>
cd ai-research-agent

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```env
# LLM — pick one
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=optional_for_openai
LLM_PROVIDER=groq

# Agent config
MAX_ITERATIONS=8
CHROMA_PERSIST_DIR=./chroma_db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

Start MLflow (optional but recommended):
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Start the API:
```bash
uvicorn app.main:app --reload --port 8000
```

---

## API

### POST /agent/run

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG and write me a minimal Python implementation",
    "max_iterations": 6
  }'
```

Response:
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is...",
  "tool_calls": [
    {"tool": "web_search", "input": "RAG LLM 2024", "output": "..."},
    {"tool": "python_executor", "input": "import chromadb...", "output": "..."}
  ],
  "iterations": 3,
  "tokens_used": 2841,
  "latency_ms": 4200,
  "mlflow_run_id": "abc123"
}
```

### GET /agent/history

Returns last 20 agent runs with question, answer, tool sequence, and token usage.

### GET /health

```json
{"status": "ok", "llm_provider": "groq", "tools_available": 4, "vector_store_docs": 127}
```

---

## MCP Server

The agent's tools are also exposed as a Model Context Protocol (MCP) server. This lets any MCP-compatible client — Claude Desktop, other LLM systems — call the same tools directly without going through the FastAPI layer.

MCP uses JSON-RPC 2.0 over stdio. The server reads requests from stdin and writes responses to stdout.

### Run standalone

```bash
python mcp_server/server.py
```

### Connect from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

Once connected, Claude Desktop can call `web_search`, `document_lookup`, `python_executor`, and `calculator` directly as MCP tools.

### How it works

The server handles three MCP methods:

- `initialize` — handshake, returns server capabilities and protocol version
- `tools/list` — returns the four available tools with their input schemas
- `tools/call` — executes the named tool with provided arguments and returns the result

Notifications (fire-and-forget messages with no `id`) are silently ignored per MCP spec.

---

## Project Structure

```
ai-research-agent/
├── app/
│   ├── main.py                  FastAPI app, lifespan, router registration
│   ├── api/
│   │   └── agent.py             /agent/run and /agent/history endpoints
│   ├── agent/
│   │   ├── graph.py             LangGraph state machine definition
│   │   ├── state.py             AgentState TypedDict
│   │   ├── nodes.py             reason(), execute_tool(), answer() nodes
│   │   └── llm_client.py       Groq/OpenAI client with one-line swap
│   ├── tools/
│   │   ├── registry.py          Tool registry — maps name → function
│   │   ├── web_search.py        DuckDuckGo search wrapper
│   │   ├── document_lookup.py   ChromaDB semantic search
│   │   ├── python_executor.py   Subprocess sandbox for code execution
│   │   └── calculator.py        Safe arithmetic evaluator
│   └── schemas/
│       ├── request.py           AgentRequest Pydantic model
│       └── response.py          AgentResponse Pydantic model
├── mcp_server/
│   └── server.py            MCP server — exposes tools over JSON-RPC stdio
├── tests/
│   ├── test_tools.py
│   ├── test_mcp_server.py
│   └── test_agent.py
├── .github/workflows/ci.yml
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```
