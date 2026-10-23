import hashlib
import json

from app.rag.loader import load_pdf
from app.rag.splitter import split_documents

PDF_PATH = "data/papers/yolo_v1.pdf"


def chunk_id(document) -> str:
    """Generate a stable identifier for a chunk."""
    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    content = document.page_content.strip()

    raw = f"{source}|{page}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def main():
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)

    output = []

    for index, document in enumerate(chunks):
        item = {
            "chunk_id": chunk_id(document),
            "chunk_index": index,
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
            "content": document.page_content,
        }
        output.append(item)

        print(f"\n{'=' * 80}")
        print(f"Chunk index: {index}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(document.page_content[:500])

    with open(
        "evaluation/chunks.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(chunks)}")
    print("Saved chunk data to evaluation/chunks.json")


if __name__ == "__main__":
    main()