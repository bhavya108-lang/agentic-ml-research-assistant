import ssl
import certifi
import urllib.parse
import urllib.request
import json

from app.tools.research_tools import ResearchTool


class DatasetSearchTool(ResearchTool):
    name = "dataset_search"

    description = (
        "Search the Hugging Face Hub for machine learning "
        "and computer vision datasets."
    )

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def run(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Dataset search query cannot be empty.")

        params = urllib.parse.urlencode(
            {
                "search": query,
                "limit": self.max_results,
            }
        )

        url = f"https://huggingface.co/api/datasets?{params}"

        ssl_context = ssl.create_default_context(
        cafile=certifi.where()
)

        with urllib.request.urlopen(
            url,
            context=ssl_context,
            timeout=15,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []

        for dataset in data:
            dataset_id = dataset.get("id", "")

            results.append(
                {
                    "name": dataset_id,
                    "url": f"https://huggingface.co/datasets/{dataset_id}",
                    "description": dataset.get(
                        "description",
                        "",
                    ),
                    "tags": dataset.get("tags", []),
                }
            )

        return results