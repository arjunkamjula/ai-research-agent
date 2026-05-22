"""
app/agent/state.py

AgentState — the single object that flows through every node
in the LangGraph state machine.

LangGraph passes this dict between nodes. Each node reads from it,
makes changes, and returns the updated state. The graph router
reads the state to decide which node runs next.
"""

from typing import TypedDict, Annotated
import operator


class ToolCall(TypedDict):
    tool:   str
    input:  str
    output: str


class AgentState(TypedDict):
    question:     str
    messages:     Annotated[list, operator.add]
    tool_calls:   Annotated[list[ToolCall], operator.add]
    iterations:   int
    final_answer: str | None
    tokens_used:  int
    error:        str | None
