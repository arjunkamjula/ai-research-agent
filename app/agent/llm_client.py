"""
app/agent/llm_client.py

LLM client supporting Groq (default) and OpenAI.
Both use the OpenAI-compatible SDK — swapping is one env var change.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o",
}


def get_client() -> OpenAI:
    if LLM_PROVIDER == "groq":
        return OpenAI(
            api_key  = os.getenv("GROQ_API_KEY"),
            base_url = "https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(
    messages:    list[dict],
    tools:       list[dict] | None = None,
    temperature: float = 0.1,
    max_tokens:  int   = 2048,
) -> dict:
    """
    Call the LLM. Returns the raw choice object as a dict.
    Supports tool_calls in response when tools are provided.
    """
    client = get_client()
    model  = MODELS.get(LLM_PROVIDER, MODELS["groq"])

    kwargs = dict(
        model       = model,
        messages    = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)

    return {
        "content":    response.choices[0].message.content,
        "tool_calls": response.choices[0].message.tool_calls,
        "model":      model,
        "provider":   LLM_PROVIDER,
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
