from typing import Any


class ResearchTool:
    """
    Base interface for external research tools.
    """

    name: str = ""
    description: str = ""

    def run(self, query: str) -> Any:
        raise NotImplementedError