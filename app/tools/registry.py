"""
app/tools/registry.py

Tool registry — maps tool names to functions and provides
OpenAI-compatible tool schemas for the LLM.
"""

from app.tools.web_search       import web_search
from app.tools.document_lookup  import document_lookup
from app.tools.python_executor  import python_executor
from app.tools.calculator       import calculator

# Map tool name → callable
TOOL_FUNCTIONS = {
    "web_search":       web_search,
    "document_lookup":  document_lookup,
    "python_executor":  python_executor,
    "calculator":       calculator,
}

# OpenAI function-calling schema — passed to the LLM
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name":        "web_search",
            "description": "Search the web for current information, news, and facts. Use when the question requires up-to-date information or external knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Search query — keep it specific and concise",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "document_lookup",
            "description": "Search the internal document vector store for relevant context. Use when looking for domain-specific knowledge or previously ingested documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Natural language query to search documents",
                    },
                    "top_k": {
                        "type":        "integer",
                        "description": "Number of results to return (default 4)",
                        "default":     4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "python_executor",
            "description": "Execute Python code and return stdout. Use for calculations, data manipulation, generating code examples, or any task requiring computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type":        "string",
                        "description": "Python code to execute. Must be self-contained. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "calculator",
            "description": "Evaluate a mathematical expression safely. Use for arithmetic, percentages, unit conversions, and simple math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type":        "string",
                        "description": "Mathematical expression to evaluate, e.g. '(42 * 1.15) / 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns string output."""
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        return str(fn(**arguments))
    except Exception as e:
        return f"Error executing {name}: {e}"
