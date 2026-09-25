from app.graph.workflow import app


def main():
    graph = app.get_graph()

    mermaid = graph.draw_mermaid()

    with open("workflow_graph.mmd", "w", encoding="utf-8") as file:
        file.write(mermaid)

    print("Workflow graph generated successfully.")
    print("Saved to: workflow_graph.mmd")


if __name__ == "__main__":
    main()