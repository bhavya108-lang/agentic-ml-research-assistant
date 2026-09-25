from app.tools.research_sources import search_research_sources


query = "traffic sign detection object detection"

sources = search_research_sources(
    query=query,
    max_results=2,
)

print("\n================ RESEARCH SOURCES ================\n")
print(sources)