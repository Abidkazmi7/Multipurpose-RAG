# Extract citations from LLM for displaying to user
def extract_citations(documents):
    citations = []

    for document in documents:
        citations.append({
            "section": document.metadata.get("Header 2"),
            "subsection": document.metadata.get("Header 3")
        })

    return citations