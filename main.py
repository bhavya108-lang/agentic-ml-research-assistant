from app.graph.workflow import app


if __name__ == "__main__":

    result = app.invoke({
        "query": "How can modern object detection models be used for traffic sign detection?",
        "research": "",
        "ml_analysis": "",
        "critique": "",
        "final_report": "",
        "next_agent": "",
        "iteration_count": 0,
    })

    print("\n\n================ FINAL REPORT ================\n")
    print(result["final_report"])