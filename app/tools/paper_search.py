import ssl
import certifi
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.tools.research_tools import ResearchTool


class PaperSearchTool(ResearchTool):
    name = "paper_search"

    description = (
        "Search arXiv for machine learning and computer vision "
        "research papers."
    )

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def run(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("Paper search query cannot be empty.")

        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
            }
        )

        url = f"https://export.arxiv.org/api/query?{params}"

        ssl_context = ssl.create_default_context(
        cafile=certifi.where()
        )

        with urllib.request.urlopen(
            url,
            context=ssl_context,
            timeout=15,
        ) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)

        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
        }

        results = []

        for entry in root.findall("atom:entry", namespace):
            title = entry.findtext(
                "atom:title",
                default="",
                namespaces=namespace,
            ).strip()

            abstract = entry.findtext(
                "atom:summary",
                default="",
                namespaces=namespace,
            ).strip()

            paper_url = entry.findtext(
                "atom:id",
                default="",
                namespaces=namespace,
            ).strip()

            published = entry.findtext(
                "atom:published",
                default="",
                namespaces=namespace,
            ).strip()

            authors = []

            for author in entry.findall(
                "atom:author",
                namespace,
            ):
                name = author.findtext(
                    "atom:name",
                    default="",
                    namespaces=namespace,
                )
                if name:
                    authors.append(name.strip())

            results.append(
                {
                    "title": title,
                    "url": paper_url,
                    "snippet": abstract,
                    "authors": authors,
                    "published": published,
                }
            )

        return results