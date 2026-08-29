from langchain_experimental.text_splitter import SemanticChunker
from models.embedding_models import huggingface_model

model = huggingface_model()

# Semantic chunking for Youtube transcript
def yt_semantic_chunk(transcript):
    chunker = SemanticChunker(model, breakpoint_threshold_type = "percentile")
    full_text = " ".join(item["text"] for item in transcript)
    chunks = chunker.create_documents([full_text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)

    return chunks