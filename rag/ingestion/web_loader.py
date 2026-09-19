import requests
from markdownify import markdownify as md
from langchain_text_splitters import MarkdownHeaderTextSplitter

def load_webpage(url):
    # Get original HTML
    response = requests.get(url)
    html = response.text

    # Convert HTML to Markdown
    markdown = md(html, heading_style='ATX')

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

    sections = markdown_splitter.split_text(markdown)

    return sections