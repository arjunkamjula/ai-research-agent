"""
app/agent/nodes.py

LangGraph node functions — each node receives AgentState,
does work, and returns a partial state update.

Three nodes:
  reason()       — LLM decides what to do next
  execute_tool() — runs the tool the LLM chose
  answer()       — synthesizes final response
"""

import json
import os
from dotenv import load_dotenv

from app.agent.state import AgentState, ToolCall
from app.agent.llm_client import chat
from app.tools.registry import TOOL_SCHEMAS, execute_tool

load_dotenv()

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))

SYSTEM_PROMPT = """You are a research agent with access to tools. 
Your goal is to answer the user's question thoroughly and accurately.

Guidelines:
- Use web_search for current information or facts you are uncertain about
- Use document_lookup to search internal knowledge base documents
- Use python_executor to run code, do calculations, or generate examples
- Use calculator for simple arithmetic
- Call tools as many times as needed — but be efficient
- After gathering enough information, synthesize a clear, well-structured answer
- Always cite your sources when using web search results
- If you have enough information to answer without tools, answer directly"""


def reason(state: AgentState) -> dict:
    """
    Reasoning node — LLM decides the next action.
    Either calls a tool or produces the final answer.

    Builds the full message history including all previous
    tool calls and their outputs so the LLM has context.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(state["messages"])

    if not state["messages"]:
        messages.append({"role": "user", "content": state["question"]})

    response = chat(
        messages = messages,
        tools    = TOOL_SCHEMAS,
    )

    new_messages = list(state["messages"])
    if not state["messages"]:
        new_messages = [{"role": "user", "content": state["question"]}]

    assistant_msg: dict = {"role": "assistant"}
    if response["content"]:
        assistant_msg["content"] = response["content"]
    if response["tool_calls"]:
        assistant_msg["tool_calls"] = [
            {
                "id":       tc.id,
                "type":     "function",
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in response["tool_calls"]
        ]

    new_messages.append(assistant_msg)

    return {
        "messages":   new_messages,
        "iterations": state["iterations"] + 1,
        "tokens_used": state["tokens_used"] + response["total_tokens"],
        "_llm_response": response,
    }


def execute_tool_node(state: AgentState) -> dict:
    """
    Tool execution node — runs the tool the LLM chose.

    Reads the tool_calls from the last assistant message,
    executes each one, and appends tool result messages
    so the LLM sees the outputs on the next iteration.
    """
    last_msg   = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])

    new_messages  = []
    new_tool_logs: list[ToolCall] = []

    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        try:
            arguments = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            arguments = {}

        output = execute_tool(tool_name, arguments)

        new_messages.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "content":      output,
        })

        new_tool_logs.append(ToolCall(
            tool   = tool_name,
            input  = json.dumps(arguments),
            output = output[:1000],
        ))

    return {
        "messages":   new_messages,
        "tool_calls": new_tool_logs,
    }


def answer(state: AgentState) -> dict:
    """
    Answer synthesis node — produces the final response.

    If the LLM already produced a text answer (no tool calls),
    uses that directly. Otherwise prompts once more to synthesize
    everything gathered into a clean final answer.
    """
    last_msg = state["messages"][-1] if state["messages"] else {}
    content  = last_msg.get("content", "")

    if content and not last_msg.get("tool_calls"):
        return {"final_answer": content}

    messages = list(state["messages"])
    messages.append({
        "role":    "user",
        "content": "Based on all the information gathered, provide a comprehensive final answer to the original question. Be clear and well-structured.",
    })

    response = chat(messages=messages, temperature=0.2)

    return {
        "final_answer": response["content"] or "Unable to generate a response.",
        "tokens_used":  state["tokens_used"] + response["total_tokens"],
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge — LangGraph router.
    Determines which node runs next based on current state.

    Returns:
        "execute" — LLM wants to call a tool
        "answer"  — LLM produced a text response or max iterations reached
    """
    if state["iterations"] >= MAX_ITERATIONS:
        return "answer"

    last_msg = state["messages"][-1] if state["messages"] else {}

    if last_msg.get("tool_calls"):
        return "execute"

    return "answer"
