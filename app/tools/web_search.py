"""
app/tools/web_search.py

DuckDuckGo search wrapper — free, no API key required.
Returns top 5 results formatted as readable text.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query:       Search query string
        max_results: Number of results to return

    Returns:
        Formatted string with titles, snippets, and URLs
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    {r.get('body', 'No snippet')}\n"
                f"    URL: {r.get('href', 'No URL')}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {e}"
