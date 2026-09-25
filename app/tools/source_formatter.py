def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "No research sources found."

    formatted = []

    for i, source in enumerate(sources, start=1):
        source_type = source.get("source_type", "unknown")
        title = source.get("title", "Unknown")
        url = source.get("url", "Unknown")
        content = source.get("content", "")
        metadata = source.get("metadata", {})

        # Keep external-source content concise to reduce LLM prompt size.
        content = content.strip()

        if len(content) > 800:
            content = content[:800] + "..."

        text = (
            f"--- Research Source {i} ---\n"
            f"Type: {source_type}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )

        if metadata:
            text += f"Metadata: {metadata}\n"

        formatted.append(text)

    return "\n".join(formatted)