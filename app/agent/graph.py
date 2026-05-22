"""
app/agent/graph.py

LangGraph state machine definition.

Graph structure:
  START → reason → [route] → execute_tool → reason (loop)
                           → answer → END

The conditional edge after reason() decides whether the LLM
wants to call a tool (loop back) or produce a final answer (end).
"""

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import reason, execute_tool_node, answer, should_continue


def build_agent_graph() -> StateGraph:
    """Build and compile the agent state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("reason",       reason)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("answer",       answer)

    graph.set_entry_point("reason")

    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "execute": "execute_tool",
            "answer":  "answer",
        },
    )

    graph.add_edge("execute_tool", "reason")
    graph.add_edge("answer",       END)

    return graph.compile()


agent_graph = build_agent_graph()


def run_agent(question: str, max_iterations: int = 8) -> dict:
    """
    Run the agent on a question and return the result.

    Args:
        question:       The user's question
        max_iterations: Safety cap on reasoning loops

    Returns:
        Dict with final_answer, tool_calls, iterations, tokens_used
    """
    import os
    os.environ["MAX_ITERATIONS"] = str(max_iterations)

    initial_state: AgentState = {
        "question":     question,
        "messages":     [],
        "tool_calls":   [],
        "iterations":   0,
        "final_answer": None,
        "tokens_used":  0,
        "error":        None,
    }

    result = agent_graph.invoke(initial_state)

    return {
        "answer":      result.get("final_answer") or "No answer generated.",
        "tool_calls":  result.get("tool_calls", []),
        "iterations":  result.get("iterations", 0),
        "tokens_used": result.get("tokens_used", 0),
    }
