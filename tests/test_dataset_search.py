from app.tools.dataset_search import DatasetSearchTool


tool = DatasetSearchTool(max_results=5)

query = "traffic sign detection"

results = tool.run(query)

print("\n================ DATASET SEARCH RESULTS ================\n")

for i, result in enumerate(results, start=1):
    print(f"========== DATASET {i} ==========")
    print("Name:", result["name"])
    print("URL:", result["url"])
    print("Description:", result["description"])
    print("Tags:", result["tags"])
    print()