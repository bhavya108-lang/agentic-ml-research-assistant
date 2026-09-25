from app.tools.web_search import WebSearchTool
from app.tools.paper_search import PaperSearchTool
from app.tools.dataset_search import DatasetSearchTool

from app.tools.source_normalizer import (
    normalize_web_result,
    normalize_paper_result,
    normalize_dataset_result,
)

from app.tools.source_formatter import format_sources


sources = []

web_result = WebSearchTool(max_results=1).run(
    "object detection computer vision"
)[0]

sources.append(normalize_web_result(web_result))


paper_result = PaperSearchTool(max_results=1).run(
    "object detection computer vision"
)[0]

sources.append(normalize_paper_result(paper_result))


dataset_result = DatasetSearchTool(max_results=1).run(
    "traffic sign detection"
)[0]

sources.append(normalize_dataset_result(dataset_result))


formatted = format_sources(sources)

print("\n================ FORMATTED RESEARCH SOURCES ================\n")
print(formatted)