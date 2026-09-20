from langchain_experimental.text_splitter import SemanticChunker
from rag.models.embedding_models import huggingface_model

# Semantic chunking for Youtube transcript
def yt_semantic_chunk(transcript, model):
    chunker = SemanticChunker(model, breakpoint_threshold_type = "percentile")
    full_text = " ".join(item["text"] for item in transcript)
    chunks = chunker.create_documents([full_text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)

    return chunks