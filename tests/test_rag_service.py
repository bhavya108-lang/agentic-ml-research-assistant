from app.rag.service import retrieve_evidence


query = "How can traffic sign detection handle difficult weather conditions?"

evidence = retrieve_evidence(query)

print("\n================ RETRIEVED EVIDENCE ================\n")
print(evidence)