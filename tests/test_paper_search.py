from app.tools.paper_search import PaperSearchTool


tool = PaperSearchTool(max_results=5)

query = "object detection computer vision"

results = tool.run(query)

print("\n================ PAPER SEARCH RESULTS ================\n")

for i, result in enumerate(results, start=1):
    print(f"========== PAPER {i} ==========")
    print("Title:", result["title"])
    print("Authors:", ", ".join(result["authors"]))
    print("Published:", result["published"])
    print("URL:", result["url"])
    print("Abstract:", result["snippet"])
    print()