from langchain_text_splitters import RecursiveCharacterTextSplitter

# Convert obtained text from PDF to chunks
def text_chunker(documents):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 1200, chunk_overlap = 300, 
    separators = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""]
    )

    splits = splitter.split_documents(documents)

    return splits