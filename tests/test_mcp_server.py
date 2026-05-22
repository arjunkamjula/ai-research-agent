"""
tests/test_mcp_server.py

Tests for the MCP server — verifies that JSON-RPC request handling
works correctly without needing a live stdio connection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import handle


def test_initialize():
    request  = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    response = handle(request)
    assert response["id"] == 1
    assert "protocolVersion" in response["result"]
    assert response["result"]["serverInfo"]["name"] == "research-agent-mcp"


def test_tools_list():
    request  = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    response = handle(request)
    tools    = response["result"]["tools"]
    names    = [t["name"] for t in tools]
    assert "web_search"      in names
    assert "document_lookup" in names
    assert "python_executor" in names
    assert "calculator"      in names


def test_tools_call_calculator():
    request = {
        "jsonrpc": "2.0",
        "id":      3,
        "method":  "tools/call",
        "params":  {"name": "calculator", "arguments": {"expression": "10 * 5"}},
    }
    response = handle(request)
    assert response["result"]["isError"] is False
    assert "50" in response["result"]["content"][0]["text"]


def test_tools_call_python_executor():
    request = {
        "jsonrpc": "2.0",
        "id":      4,
        "method":  "tools/call",
        "params":  {"name": "python_executor", "arguments": {"code": "print(2 + 2)"}},
    }
    response = handle(request)
    assert response["result"]["isError"] is False
    assert "4" in response["result"]["content"][0]["text"]


def test_unknown_tool():
    request = {
        "jsonrpc": "2.0",
        "id":      5,
        "method":  "tools/call",
        "params":  {"name": "nonexistent_tool", "arguments": {}},
    }
    response = handle(request)
    assert "error" in response


def test_notification_returns_none():
    request  = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    response = handle(request)
    assert response is None
