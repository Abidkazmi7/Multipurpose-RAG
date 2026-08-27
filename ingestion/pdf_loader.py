import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

def load_pdf(pdf_path):
    # Convert entire PDF into one Markdown document
    markdown = pymupdf4llm.to_markdown(pdf_path)

    # Headers that define sections
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Split the entire Markdown document into sections
    sections = markdown_splitter.split_text(markdown)

    return sections