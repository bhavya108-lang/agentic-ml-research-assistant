from app.tools.web_search import WebSearchTool
from app.tools.paper_search import PaperSearchTool
from app.tools.dataset_search import DatasetSearchTool

from app.tools.source_normalizer import (
    normalize_web_result,
    normalize_paper_result,
    normalize_dataset_result,
)

from app.tools.source_formatter import format_sources


def search_research_sources(
    query: str,
    max_results: int = 3,
) -> str:

    sources = []

    # Web search
    try:
        web_results = WebSearchTool(
            max_results=max_results
        ).run(query)
    except Exception as exc:
        print(f"Web search failed: {exc}")
        web_results = []

    for result in web_results:
        sources.append(
            normalize_web_result(result)
        )

    # Paper search
    try:
        paper_results = PaperSearchTool(
            max_results=max_results
        ).run(query)
    except Exception as exc:
        print(f"Paper search failed: {exc}")
        paper_results = []

    for result in paper_results:
        sources.append(
            normalize_paper_result(result)
        )

    # Dataset search
    try:
        dataset_results = DatasetSearchTool(
            max_results=max_results
        ).run(query)
    except Exception as exc:
        print(f"Dataset search failed: {exc}")
        dataset_results = []

    for result in dataset_results:
        sources.append(
            normalize_dataset_result(result)
        )

    return format_sources(sources)