"""
mcp_server/server.py

Model Context Protocol (MCP) server that exposes the agent's tools
as MCP-compatible resources. This lets any MCP-capable client
(Claude Desktop, other LLMs) call web_search, document_lookup,
python_executor, and calculator directly over the MCP protocol.

MCP uses JSON-RPC 2.0 over stdio. The server reads requests from
stdin and writes responses to stdout — standard MCP transport.

How MCP works:
  The client sends a tools/list request to discover available tools.
  The client sends a tools/call request with a tool name and arguments.
  The server executes the tool and returns the result.
  All communication is newline-delimited JSON over stdio.

Run standalone:
    python mcp_server/server.py

Connect from Claude Desktop by adding to claude_desktop_config.json:
    {
      "mcpServers": {
        "research-agent": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server/server.py"]
        }
      }
    }
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.web_search      import web_search
from app.tools.document_lookup import document_lookup
from app.tools.python_executor import python_executor
from app.tools.calculator      import calculator

MCP_TOOLS = [
    {
        "name":        "web_search",
        "description": "Search the web using DuckDuckGo. Returns top 5 results with titles, snippets, and URLs. Use for current information, news, and facts.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Search query — keep it specific and concise",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name":        "document_lookup",
        "description": "Search the local ChromaDB vector store for relevant documents. Use for domain-specific knowledge and previously ingested content.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Natural language query",
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
    {
        "name":        "python_executor",
        "description": "Execute Python code in a sandboxed subprocess and return stdout. Use for calculations, data manipulation, and generating code examples.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "code": {
                    "type":        "string",
                    "description": "Python code to execute. Use print() to output results.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name":        "calculator",
        "description": "Safely evaluate a mathematical expression. Use for arithmetic, percentages, and unit conversions.",
        "inputSchema": {
            "type":       "object",
            "properties": {
                "expression": {
                    "type":        "string",
                    "description": "Mathematical expression e.g. '(42 * 1.15) / 3'",
                }
            },
            "required": ["expression"],
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search":      web_search,
    "document_lookup": document_lookup,
    "python_executor": python_executor,
    "calculator":      calculator,
}


def send(response: dict) -> None:
    """Write a JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def handle(request: dict) -> dict | None:
    """
    Handle a single JSON-RPC request.

    MCP methods:
      initialize          — handshake, return server capabilities
      tools/list          — return available tools
      tools/call          — execute a tool
      notifications/*     — fire-and-forget, no response needed
    """
    req_id  = request.get("id")
    method  = request.get("method", "")
    params  = request.get("params", {})

    # Notifications have no id — no response required
    if req_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id":      req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo": {
                    "name":    "research-agent-mcp",
                    "version": "1.0.0",
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id":      req_id,
            "result":  {"tools": MCP_TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        fn = TOOL_FUNCTIONS.get(tool_name)
        if not fn:
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "error": {
                    "code":    -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            }

        try:
            result = fn(**arguments)
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "result": {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Tool error: {e}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id":      req_id,
        "error": {
            "code":    -32601,
            "message": f"Method not found: {method}",
        },
    }


def main() -> None:
    """Read JSON-RPC requests from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request  = json.loads(line)
            response = handle(request)
            if response is not None:
                send(response)
        except json.JSONDecodeError:
            send({
                "jsonrpc": "2.0",
                "id":      None,
                "error":   {"code": -32700, "message": "Parse error"},
            })
        except Exception as e:
            send({
                "jsonrpc": "2.0",
                "id":      None,
                "error":   {"code": -32603, "message": f"Internal error: {e}"},
            })


if __name__ == "__main__":
    main()
