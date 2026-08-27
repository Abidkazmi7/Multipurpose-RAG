import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Convert obtained text from PDF to chunks
def text_chunker(sections):
    # Parent splitter
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 2000, 
        chunk_overlap = 200, 
        separators = ["\n\n", "\n", ". ", " ", ""]
    )

    # Child splitter
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 400, 
        chunk_overlap = 50, 
        separators = ["\n\n", "\n", ". ", " ", ""]
    )

    parents = parent_splitter.split_documents(sections)

    # Lists for storing each chunk type
    parent_documents = []
    child_documents = []

    # Loop through parent chunks
    for parent in parents:
        # Generate ID for parent chunk
        parent_id = str(uuid.uuid4())

        # Add ID to chunk metadata
        parent.metadata["parent_id"] = parent_id
        parent.metadata["chunk_type"] = "parent"

        parent_documents.append(parent)

        child_chunks = child_splitter.split_documents([parent])

        # Loop through child chunks
        for child in child_chunks:
            child_id = str(uuid.uuid4())

            child.metadata["parent_id"] = parent_id
            child.metadata["child_id"] = child_id
            child.metadata["chunk_type"] = "child"

            child_documents.append(child)

    return parent_documents, child_documents