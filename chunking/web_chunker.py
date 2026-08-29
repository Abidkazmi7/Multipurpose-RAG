import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def webpage_chunker(sections):

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(sections)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_type"] = "web"

    return chunks