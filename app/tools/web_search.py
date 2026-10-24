import os
import time
import functools
from duckduckgo_search import DDGS
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential
from app.tools.research_tools import ResearchTool

# 1. DDGS Primary (Free, but flaky and snippet-only)
@functools.lru_cache(maxsize=50)
def _ddgs_search(query: str, max_results: int) -> list[dict]:
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4), reraise=True)
    def _execute():
        with DDGS(timeout=10) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    return _execute()

# 2. Tavily Fallback (Requires API key, but highly reliable with rich content)
@functools.lru_cache(maxsize=50)
def _tavily_search(query: str, max_results: int) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[WebSearchTool] TAVILY_API_KEY missing. Skipping fallback.")
        return []
        
    client = TavilyClient(api_key=api_key)
    # include_raw_content=True pulls actual page text instead of just SEO snippets
    response = client.search(query=query, max_results=max_results, include_raw_content=True)
    
    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
            # Prefer raw content for better RAG grounding, fall back to snippet
            "snippet": item.get("raw_content", "") or item.get("content", ""),
        })
    return results

class WebSearchTool(ResearchTool):
    name = "web_search"
    description = (
        "Search the web for information relevant to "
        "machine learning and computer vision research."
    )

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._internal_fetch_count = max_results * 2

    def run(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
            
        # Attempt Primary: DuckDuckGo
        try:
            raw_results = _ddgs_search(query, self._internal_fetch_count)
            if raw_results:
                results = []
                for result in raw_results[:self.max_results]:
                    snippet = result.get("body", "").strip()
                    if snippet:
                        results.append({
                            "title": result.get("title", "Untitled"),
                            "url": result.get("href", ""),
                            "snippet": snippet,
                        })
                if results:
                    return results
        except Exception as e:
            print(f"[WebSearchTool] DDGS Primary failed ({e}). Routing to Tavily Fallback.")

        # Attempt Fallback: Tavily
        try:
            print(f"[WebSearchTool] Executing Tavily search for: '{query}'")
            return _tavily_search(query, self.max_results)
        except Exception as e:
            print(f"[WebSearchTool] Tavily Fallback failed: {e}")
            return []