import re
import pymupdf4llm

from ingestion.pdf_loader import load_pdf

# Find out what headers belong to what page number
def detect_headers_by_page(pdf_path):
    page_chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True
    )

    header_pages = []

    for page_chunk in page_chunks:
        page_number = page_chunk["metadata"]["page_number"]
        page_text = page_chunk["text"]

        for line in page_text.splitlines():
            line = line.strip()

            # Detect Markdown headers from # to ####
            match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)

            if not match:
                continue

            header_level = len(match.group(1))
            header_text = match.group(2)

            # Remove Markdown bold markers, e.g: **1 Introduction** -> 1 Introduction
            header_text = re.sub(r"\*\*(.*?)\*\*", r"\1", header_text)

            header_pages.append({
                "header": header_text,
                "level": header_level,
                "page": page_number
            })

    return header_pages

# Add page metadata to each section
def attach_page_metadata(sections, pdf_path):
    header_pages = detect_headers_by_page(pdf_path)

    for section in sections:
        # Get the deepest header available for this section
        section_header = next(
            (
                value
                for key, value in reversed(section.metadata.items())
                if key.startswith("Header")
            ),
            None
        )

        if section_header is None:
            continue

        section_header = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            section_header
        )

        # Find the page where this section starts
        for item in header_pages:
            if item["header"] == section_header:
                section.metadata["start_page"] = item["page"]
                break

    return sections

# Extract citations from LLM for displaying to user
def extract_citations(documents):
    citations = []

    for document in documents:
        citations.append({
            "page": document.metadata.get("start_page"),
            "section": document.metadata.get("Header 2"),
            "subsection": document.metadata.get("Header 3")
        })

    return citations