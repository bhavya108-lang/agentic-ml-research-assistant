from app.tools.web_search import WebSearchTool


tool = WebSearchTool()

query = "recent computer vision object detection research"

results = tool.run(query)

print("\n================ WEB SEARCH RESULTS ================\n")

for i, result in enumerate(results, start=1):
    print(f"========== RESULT {i} ==========")
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Snippet:", result["snippet"])
    print()