import time
import urllib.error
from app.tools.web_search import WebSearchTool
from app.tools.paper_search import PaperSearchTool
from app.tools.dataset_search import DatasetSearchTool

def evaluate_tools(queries: list[str], max_results: int = 3) -> dict:
    tools = {
        "web_search": WebSearchTool(max_results=max_results),
        "paper_search": PaperSearchTool(max_results=max_results),
        "dataset_search": DatasetSearchTool(max_results=max_results),
    }
    report = {}

    for name, tool in tools.items():
        successes = 0
        empty_returns = 0
        timeouts_or_failures = 0
        latencies = []

        for q in queries:
            start = time.perf_counter()
            try:
                res = tool.run(q)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                if isinstance(res, list) and len(res) > 0:
                    successes += 1
                else:
                    empty_returns += 1  # HTTP 200 but 0 results (e.g. HF query too long)
            except (TimeoutError, urllib.error.URLError, Exception):
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                timeouts_or_failures += 1

        total = len(queries)
        report[name] = {
            "success_rate": round(successes / total, 4),
            "empty_result_rate": round(empty_returns / total, 4),
            "failure_or_timeout_rate": round(timeouts_or_failures / total, 4),
            "avg_latency_sec": round(sum(latencies) / total, 3) if latencies else 0.0,
        }
    return report