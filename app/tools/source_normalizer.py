def normalize_web_result(result: dict) -> dict:
    return {
        "source_type": "web",
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "content": result.get("snippet", ""),
        "metadata": {},
    }


def normalize_paper_result(result: dict) -> dict:
    return {
        "source_type": "paper",
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "content": result.get("snippet", ""),
        "metadata": {
            "authors": result.get("authors", []),
            "published": result.get("published", ""),
        },
    }


def normalize_dataset_result(result: dict) -> dict:
    return {
        "source_type": "dataset",
        "title": result.get("name", ""),
        "url": result.get("url", ""),
        "content": result.get("description", ""),
        "metadata": {
            "tags": result.get("tags", []),
            "task_categories": result.get("task_categories", []),
            "downloads": result.get("downloads", 0),
            "likes": result.get("likes", 0),
        },
    }