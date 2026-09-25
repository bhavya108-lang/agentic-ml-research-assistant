from app.tools.web_search import WebSearchTool
from app.tools.paper_search import PaperSearchTool
from app.tools.dataset_search import DatasetSearchTool

from app.tools.source_normalizer import (
    normalize_web_result,
    normalize_paper_result,
    normalize_dataset_result,
)


print("\n================ WEB SOURCE ================\n")

web_results = WebSearchTool(max_results=1).run(
    "object detection computer vision"
)

web_source = normalize_web_result(web_results[0])

print(web_source)


print("\n================ PAPER SOURCE ================\n")

paper_results = PaperSearchTool(max_results=1).run(
    "object detection computer vision"
)

paper_source = normalize_paper_result(paper_results[0])

print(paper_source)


print("\n================ DATASET SOURCE ================\n")

dataset_results = DatasetSearchTool(max_results=1).run(
    "traffic sign detection"
)

dataset_source = normalize_dataset_result(dataset_results[0])

print(dataset_source)