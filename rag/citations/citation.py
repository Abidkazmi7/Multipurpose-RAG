# Extract citations from LLM for displaying to user
def extract_citations(documents):
    citations = []

    for document in documents:
        section = document.metadata.get("Header 2")
        subsection = document.metadata.get("Header 3")

        if section or subsection:
            citations.append({
                "section": section,
                "subsection": subsection
            })

    return citations