from ddgs import DDGS

from app.tools.research_tools import ResearchTool


class WebSearchTool(ResearchTool):
    name = "web_search"

    description = (
        "Search the web for information relevant to "
        "machine learning and computer vision research."
    )

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def run(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=self.max_results,
            )

            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    }
                )

        return results