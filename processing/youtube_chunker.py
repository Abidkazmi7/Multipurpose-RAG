from ingestion.youtube_loader import load_youtube
from langchain_experimental.text_splitter import SemanticChunker
from embeddings.models import huggingface_model

model = huggingface_model()

# Semantic chunking for Youtube transcript
def yt_semantic_chunk(transcript):
    chunker = SemanticChunker(model, breakpoint_threshold_type = "percentile")
    full_text = " ".join(item["text"] for item in transcript)
    chunks = chunker.create_documents([full_text])

    return chunks